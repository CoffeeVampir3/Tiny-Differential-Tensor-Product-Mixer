import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

def count_parameters_layerwise(model):
    # Layerwise params, turn this into a util function.
    total_params = 0
    layer_params = {}
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
            
        param_count = parameter.numel()
        layer_params[name] = param_count
        total_params += param_count
    
    print(f"\nModel Parameter Summary:")
    print("-" * 60)
    for name, count in layer_params.items():
        print(f"{name}: {count:,} parameters")
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params:,}\n")
    
    return total_params

def save_checkpoint(model, optimizer, filename="checkpoint.safetensors"):
    # Get the original uncompiled model if it's compiled
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()
    
    save_file(model_state, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.safetensors"):
    model_state = load_file(filename)
    
    # Load state dict into the original uncompiled model if it's compiled
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)