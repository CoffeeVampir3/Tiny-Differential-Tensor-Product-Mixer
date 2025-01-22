import torch
import torch.nn as nn
import torch.nn.functional as F

from .extact import xATGLU
from .tpa import TensorProductTransformerBlock
from .diff_attn import DifferentialAttentionBlock

class WaifuLMUwU(nn.Module):
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=8, max_seq_length=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        blocks = []
        for i in range(n_layer//2):
            blocks.append(TensorProductTransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                head_dim=n_embd // n_head,
                rank=2,
                q_rank=6,
                using_groupnorm=True,
                use_rope=True
            ))
            
            blocks.append(DifferentialAttentionBlock(
                n_embd=n_embd,
                n_head=n_head,
                head_dim=n_embd // n_head,
                layer_num=i+1,
                using_groupnorm=True,
                use_rope=False
            ))

        self.blocks = nn.ModuleList(blocks)
        
        self.ln_f = nn.LayerNorm(n_embd)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        x = self.token_embedding(idx)
        
        # batch seq dim
        # x is [4, 1024, 768] here
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # We don't need an LM head, this is essentially the same idea as weight tying -- we reuse the input classifier 
        
        # For CCE:
        # x -> The normalized hidden states
        # head weight -> The classifier weight matrix
        return x, self.token_embedding.weight