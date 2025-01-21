import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor
import math
from .extact import xATGLU

# Tensor Product Attention
# https://arxiv.org/abs/2501.06425

# The starting point for this code is from the Tensor Product Attention github repo:
# Modified, original from https://github.com/tensorgi/T6/blob/main/model/T6_ropek.py

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim/4).float() / dim/4))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()[None, :, None, :].permute(0, 2, 1, 3).bfloat16()
            self.sin_cached = freqs.sin()[None, :, None, :].permute(0, 2, 1, 3).bfloat16()
        return self.cos_cached, self.sin_cached

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin 
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1)

class DifferentialAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, layer_num, using_groupnorm=True, use_rope=True):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.use_rope = use_rope
        
        # Projections
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd * 2, bias=False)
        
        # Lambda parameters
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_num - 1))
        
        # Normalization
        if using_groupnorm:
            self.norm = nn.GroupNorm(
                num_groups=n_head, 
                num_channels=n_head * 2 * head_dim, 
                eps=1e-5, 
                affine=False
            )
        
        self.output_projection =xATGLU(n_embd * 2, n_embd, bias=False)
        
        if use_rope:
            self.rotary = Rotary(head_dim)

    def _split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_head, -1)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_dim = x.size()
        return x.reshape(batch, num_heads * head_dim, seq_len)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        # batch seq embd/2
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)
        
        # batch seq embd/2 -> batch n_head seq dim
        Q1 = self._split_heads(Q1)
        Q2 = self._split_heads(Q2)
        K1 = self._split_heads(K1)
        K2 = self._split_heads(K2)
        V = self._split_heads(V) # dim*2
        
        if self.use_rope:
            cos, sin = self.rotary(x)
            Q1 = apply_rotary_emb(Q1, cos, sin)
            Q2 = apply_rotary_emb(Q2, cos, sin)
            K1 = apply_rotary_emb(K1, cos, sin)
            K2 = apply_rotary_emb(K2, cos, sin)
        
        Q1 = Q1 * (self.head_dim ** -0.5)
        Q2 = Q2 * (self.head_dim ** -0.5)
        
        lambda_value = (torch.exp(self.lambda_q1 @ self.lambda_k1) - 
                       torch.exp(self.lambda_q2 @ self.lambda_k2) + 
                       self.lambda_init)
        
        # [batch, n_head, seq_len, seq_len]
        attn1 = torch.matmul(Q1, K1.transpose(-2, -1))
        attn2 = torch.matmul(Q2, K2.transpose(-2, -1))
        
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        attn = attn1 - lambda_value * attn2
        
        # batch n_head seq dim*2
        out = torch.matmul(attn, V)
        
        # [batch, n_embd * 2, seq_len]
        out = self._merge_heads(out)
        
        # [batch, n_embd * 2, seq_len] -> # [batch, seq_len, n_embd * 2]
        out = self.norm(out).permute(0, 2, 1)
        
        out = out * (1 - self.lambda_init)
        
        out = self.output_projection(out)
        
        return out
    
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * n_embd)
        
        self.fc = xATGLU(n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.fc(x)
        x = self.c_proj(x)
        return x

class DifferentialAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, layer_num, using_groupnorm=True, use_rope=True):
        super().__init__()
        self.attn = DifferentialAttention(n_embd, n_head, head_dim, layer_num, using_groupnorm, use_rope)
        self.mlp = MLP(n_embd)
        self.learned_residual_scale_attn = nn.Parameter(torch.ones(1))
        self.learned_residual_scale_mlp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Single pre-normalization at the start
        x_normed = F.rms_norm(x, (x.size(-1),))
        x = self.learned_residual_scale_attn * x + self.attn(x_normed)
        x = self.learned_residual_scale_mlp * x + self.mlp(x)
        return x