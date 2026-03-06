#!/usr/bin/env python3
# just to play around and understand
import argparse
import os
import re
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from msc import evaluate_epoch, FashionMnistIdxDataset, get_device
from mlp_ib import VIBNet

torch.set_printoptions(
    threshold=float("inf"),
    linewidth=200,
    precision=4,
    sci_mode=False,
)
os.makedirs("plots", exist_ok=True)

o_shape = 10
i_shape = 784
parser = argparse.ArgumentParser(description="inspect and prune saved mlp vib model layers")
parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help="directory under save_stats_weights containing a trained run",
)
args = parser.parse_args()
run_dir = args.run_dir.rstrip("/")
if not os.path.isdir(run_dir):
    save_weights_path = os.path.join("save_stats_weights", run_dir)
    if os.path.isdir(save_weights_path):
        run_dir = save_weights_path
    else:
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

run_name = os.path.basename(run_dir)
match = re.match(r"^vib_(mlp|cnn)_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)(?:_(\d+))?$", run_name)
if not match:
    raise ValueError(
        f"run_dir name must match: vib_<model>_<hidden1>_<hidden2>_<z_dim>_<beta>_<lr>[_<epochs>], got: {run_name}"
    )
model_name, h1_s, h2_s, z_dim_s, beta_s, _, _ = match.groups()
if model_name != "mlp":
    raise ValueError(f"this script currently supports mlp runs only, got model: {model_name}")
h1 = int(h1_s)
h2 = int(h2_s)
z_dim = int(z_dim_s)
beta = float(beta_s)

pth_candidates = sorted([f for f in os.listdir(run_dir) if f.endswith(".pth")])
if not pth_candidates:
    raise FileNotFoundError(f"no .pth file found in: {run_dir}")
weights_path = os.path.join(run_dir, pth_candidates[0])
weights = torch.load(weights_path, map_location="cpu")
model = VIBNet(z_dim, i_shape, h1, h2, o_shape)
model.load_state_dict(weights)


# ---------- inspecting weight matrices via heat map ----------
def plot_weight_heatmaps(model: nn.Module, file_name: str, cmap: str = "viridis") -> None:
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not layers:
        return
    n = len(layers)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (name, module) in enumerate(layers):
        matrix = module.weight.detach().cpu().numpy()
        im = axes[i].imshow(matrix, aspect="auto", cmap=cmap)
        total_params = sum(p.numel() for p in module.parameters())
        axes[i].set_title(f"{name} ({total_params} params)")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}_weight_heat_map.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


plot_weight_heatmaps(model, run_name)
