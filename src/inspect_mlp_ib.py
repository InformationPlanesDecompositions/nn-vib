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
from copy import deepcopy
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
    "run_dir name must match: vib_<model>_<hidden1>_<hidden2>_<z_dim>_<beta>_<lr>[_<epochs>], "
    f"got: {run_name}"
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

# ---------- inspecting neurons ----------
def indices_above_avg_abs_threshold(
  model: nn.Module,
  layer_name: str,
  top_k: int,
  axis: str = "row"
) -> dict[int, float]:
  layer = dict(model.named_modules())[layer_name]
  weights = layer.weight.detach()
  avg_abs_by_idx = []
  if axis == "row":
    for row_idx in range(weights.shape[0]):
      avg_abs_weight = weights[row_idx, :].abs().mean().item()
      avg_abs_by_idx.append((row_idx, avg_abs_weight))
  elif axis == "col":
    for col_idx in range(weights.shape[1]):
      avg_abs_weight = weights[:, col_idx].abs().mean().item()
      avg_abs_by_idx.append((col_idx, avg_abs_weight))
  else:
    raise ValueError("axis must be 'row' or 'col'")
  avg_abs_by_idx.sort(key=lambda x: x[1], reverse=True)
  top_pairs = avg_abs_by_idx[:top_k]
  top_dict = {idx: avg for idx, avg in top_pairs}
  for idx, avg in top_pairs: print(f"{idx}: {avg:.6f}")
  return top_dict

layers = ["fc_mu", "fc_logvar"]
n = 10
to_prune_neurons = {}
for layer in layers:
  indices = indices_above_avg_abs_threshold(model, layer, n, axis="row")
  to_prune_neurons[layer] = list(indices.keys())
  print("---")

print(to_prune_neurons)

# ---------- full network magnitude pruning ----------
def magnitude_prune_top_percent(model: nn.Module, p: float) -> None:
  for name, module in model.named_modules():
    if name in ["fc_decode", "fc1"]: continue
    if isinstance(module, nn.Linear):
      prune.l1_unstructured(module, name="weight", amount=p)
      prune.remove(module, "weight")

device = get_device()
test_loader = DataLoader(
  FashionMnistIdxDataset("data/mnist_fashion/", train=False),
  batch_size=100,
  shuffle=False
)
base_loss, _, _, base_acc = evaluate_epoch(model.to(device), test_loader, device, beta=beta)
print(f"before prune: loss={base_loss:.6f}, acc={base_acc:.2f}")
for pct in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]:
  pruned_model = VIBNet(z_dim, i_shape, h1, h2, o_shape).to(device)
  pruned_model.load_state_dict(weights)
  magnitude_prune_top_percent(pruned_model, pct)
  loss, _, _, acc = evaluate_epoch(pruned_model, test_loader, device, beta=beta)
  print(f"after prune {int(pct*100)}%: loss={loss:.6f}, acc={acc:.2f}")

# ---------- prune neurons ----------
def mask_keep_neurons(
  weight_matrix: torch.Tensor,
  keep_row_indices: list[int] | None = None,
  keep_col_indices: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  mask = torch.zeros_like(weight_matrix)
  if keep_row_indices is None and keep_col_indices is None:
    return weight_matrix * mask, mask
  if keep_row_indices is None:
    keep_row_indices = list(range(weight_matrix.shape[0]))
  if keep_col_indices is None:
    keep_col_indices = list(range(weight_matrix.shape[1]))
  row_idx = torch.tensor(keep_row_indices, device=weight_matrix.device)
  col_idx = torch.tensor(keep_col_indices, device=weight_matrix.device)
  mask[row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = 1
  return weight_matrix * mask, mask

dict_pruned_model = deepcopy(model)
with torch.no_grad():
  for layer_name, keep_rows in to_prune_neurons.items():
    layer = dict(dict_pruned_model.named_modules()).get(layer_name)
    if layer is None:
      raise ValueError(f"layer '{layer_name}' not found in model")
    if not isinstance(layer, nn.Linear):
      raise ValueError(f"layer '{layer_name}' is not nn.Linear")
    pruned_weights, _ = mask_keep_neurons(
      layer.weight,
      keep_row_indices=keep_rows
    )
    layer.weight.copy_(pruned_weights)

pruned_loss, _, _, pruned_acc = evaluate_epoch(
  dict_pruned_model.to(device),
  test_loader,
  device,
  beta=beta
)
print(f"dict prune: loss={pruned_loss:.6f}, acc={pruned_acc:.2f}")

input("press enter to terminate...") # to keep any plots still open
