from typing import List, Dict
import torch
import torch.nn as nn

def svd(model: nn.Module, layers: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    results = {}

    named_modules = dict(model.named_modules())

    for layer_name in layers:
        if layer_name not in named_modules:
            print(f"warning: layer '{layer_name}' not found in model!")
            continue

        layer = named_modules[layer_name]

        if hasattr(layer, "weight"):
            weights = layer.weight.data.detach().float()
            if weights.dim() > 2:
                weights = weights.view(weights.size(0), -1)

            # W = U Σ V^T
            U, S, Vh = torch.linalg.svd(weights, full_matrices=False)

            results[layer_name] = {
                "U": U, # output basis
                "S": S, # singular values
                "Vh": Vh, # input basis
            }
        else:
            print(f"warning: layer '{layer_name}' does not have a 'weight' attribute!")
    return results
