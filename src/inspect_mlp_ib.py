#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass

import matplotlib

if "MPLBACKEND" not in os.environ:
  matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlp_ib import VIBNet
from msc import FashionMnistIdxDataset, evaluate_epoch, get_device

# run selection
target_lr = 0.0002
target_epochs = 400
input_shape = 784
output_shape = 10
batch_size = 128

# pruning experiment config
layers_to_prune = ["fc_mu", "fc_logvar", "fc2"]
layers_to_prune_fc2_only = ["fc2"]
prune_layer_sets = [layers_to_prune, layers_to_prune_fc2_only]
layers_to_inspect = ["fc_mu", "fc_logvar"]
top_k_neurons = 10
prune_percents = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# paths
save_root = "save_stats_weights"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)


@dataclass
class RunSpec:
  run_dir: str
  run_name: str
  beta: float
  lr: float
  epochs: int
  seed: int


def indices_above_avg_abs_threshold(
  model: nn.Module,
  layer_name: str,
  top_k: int,
  axis: str = "row",
) -> dict[int, float]:
  layer = dict(model.named_modules())[layer_name]
  if not isinstance(layer, nn.Linear):
    raise ValueError(f"layer is not nn.Linear: {layer_name}")
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
  return {idx: avg for idx, avg in top_pairs}


def inspect_top_neurons(
  model: nn.Module,
  layer_names: list[str],
  top_k: int,
) -> dict[str, list[int]]:
  top_neurons = {}
  for layer_name in layer_names:
    indices = indices_above_avg_abs_threshold(model, layer_name, top_k, axis="row")
    top_neurons[layer_name] = list(indices.keys())
    print(f"  inspect {layer_name}: {top_neurons[layer_name]}")
  return top_neurons


def mask_keep_neurons(
  weight_matrix: torch.Tensor,
  keep_row_indices: list[int] | None = None,
  keep_col_indices: list[int] | None = None,
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


def neuron_prune_layers(model: nn.Module, layer_names: list[str], amount: float) -> None:
  if amount <= 0:
    return
  modules = dict(model.named_modules())
  for layer_name in layer_names:
    module = modules.get(layer_name)
    if module is None:
      raise ValueError(f"layer not found in model: {layer_name}")
    if not isinstance(module, nn.Linear):
      raise ValueError(f"layer is not nn.Linear: {layer_name}")
    row_count = module.weight.shape[0]
    keep_count = int(round((1.0 - amount) * row_count))
    keep_count = max(0, min(row_count, keep_count))
    if keep_count == row_count:
      continue
    avg_abs_by_row = []
    for row_idx in range(row_count):
      avg_abs_weight = module.weight[row_idx, :].detach().abs().mean().item()
      avg_abs_by_row.append((row_idx, avg_abs_weight))
    avg_abs_by_row.sort(key=lambda x: x[1], reverse=True)
    keep_rows = [idx for idx, _ in avg_abs_by_row[:keep_count]]
    with torch.no_grad():
      pruned_weights, _ = mask_keep_neurons(module.weight, keep_row_indices=keep_rows)
      module.weight.copy_(pruned_weights)
      if module.bias is not None:
        bias_mask = torch.zeros_like(module.bias)
        if keep_rows:
          keep_idx = torch.tensor(keep_rows, device=module.bias.device)
          bias_mask[keep_idx] = 1
        module.bias.mul_(bias_mask)


def parse_run_specs(
  root_dir: str,
  target_hidden1: int,
  target_hidden2: int,
  target_z_dim: int,
  target_seed: int,
  lr_filter: float | None,
  epochs_filter: int | None,
) -> list[RunSpec]:
  pattern = re.compile(r"^vib_mlp_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$")
  candidates = []
  for run_name in os.listdir(root_dir):
    run_dir = os.path.join(root_dir, run_name)
    if not os.path.isdir(run_dir):
      continue
    match = pattern.match(run_name)
    if not match:
      continue
    h1_s, h2_s, z_s, beta_s, lr_s, epochs_s, seed_s = match.groups()
    if int(h1_s) != target_hidden1 or int(h2_s) != target_hidden2 or int(z_s) != target_z_dim:
      continue
    if int(seed_s) != target_seed:
      continue
    lr_value = float(lr_s)
    epochs_value = int(epochs_s)
    if lr_filter is not None and lr_value != lr_filter:
      continue
    if epochs_filter is not None and epochs_value != epochs_filter:
      continue
    candidates.append(
      RunSpec(
        run_dir=run_dir,
        run_name=run_name,
        beta=float(beta_s),
        lr=lr_value,
        epochs=epochs_value,
        seed=int(seed_s),
      )
    )
  if not candidates:
    return []
  selected_by_beta = {}
  for run in candidates:
    prev = selected_by_beta.get(run.beta)
    if prev is None:
      selected_by_beta[run.beta] = run
      continue
    if run.epochs > prev.epochs:
      selected_by_beta[run.beta] = run
  return sorted(selected_by_beta.values(), key=lambda r: r.beta)


def load_state_dict_from_run(run_dir: str) -> dict[str, torch.Tensor]:
  pth_files = sorted([name for name in os.listdir(run_dir) if name.endswith(".pth")])
  if not pth_files:
    raise FileNotFoundError(f"no .pth file found in {run_dir}")
  return torch.load(os.path.join(run_dir, pth_files[0]), map_location="cpu")


def run_pruning_stability(
  runs: list[RunSpec],
  hidden1: int,
  hidden2: int,
  z_dim: int,
  layer_names: list[str],
  prune_fn,
  prune_percent_values: list[int],
) -> dict[float, tuple[list[int], list[float]]]:
  device = get_device()
  test_loader = DataLoader(
    FashionMnistIdxDataset("data/mnist_fashion/", train=False), batch_size=batch_size, shuffle=False
  )
  curves = {}
  for run in runs:
    print(f"\nrun: {run.run_name}")
    state_dict = load_state_dict_from_run(run.run_dir)
    # inspect_model = VIBNet(z_dim, input_shape, hidden1, hidden2, output_shape).to(device)
    # inspect_model.load_state_dict(state_dict)
    # inspect_top_neurons(inspect_model, layers_to_inspect, top_k_neurons)
    xs = []
    ys = []
    for pct in prune_percent_values:
      amount = pct / 100.0
      model = VIBNet(z_dim, input_shape, hidden1, hidden2, output_shape).to(device)
      model.load_state_dict(state_dict)
      prune_fn(model, layer_names, amount)
      loss, _, _, acc = evaluate_epoch(model, test_loader, device, beta=run.beta)
      xs.append(pct)
      ys.append(loss)
      print(f"  prune={pct:>3}% loss={loss:.6f} acc={acc:.2f}")
    curves[run.beta] = (xs, ys)
  return curves


def plot_curves(
  curves: dict[float, tuple[list[int], list[float]]],
  hidden1: int,
  hidden2: int,
  z_dim: int,
  seed: int,
  layer_names: list[str],
) -> str:
  plt.figure(figsize=(10, 6))
  for beta, (xs, ys) in sorted(curves.items(), key=lambda item: item[0]):
    plt.plot(xs, ys, marker="o", linewidth=2, label=f"beta={beta:g}")
  layer_part = ", ".join(layer_names)
  plt.title(f"mlp ({hidden1}, {hidden2}, {z_dim}) seed={seed} | pruned: {layer_part}")
  plt.xlabel("percent of neurons pruned")
  plt.ylabel("test loss")
  plt.grid(True, alpha=0.3)
  plt.legend()
  layer_key = "_".join(layer_names)
  save_name = f"mlp_prune_h1_{hidden1}_h2_{hidden2}_z_{z_dim}_seed_{seed}_{layer_key}.png"
  save_path = os.path.join(plots_dir, save_name)
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  return save_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="inspect mlp pruning stability across betas")
  parser.add_argument("--hidden1", type=int, required=True, help="first hidden layer width")
  parser.add_argument("--hidden2", type=int, required=True, help="second hidden layer width")
  parser.add_argument("--z_dim", type=int, required=True, help="latent bottleneck size")
  parser.add_argument("--seed", type=int, required=True, help="random seed")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  hidden1 = args.hidden1
  hidden2 = args.hidden2
  z_dim = args.z_dim
  seed = args.seed
  runs = parse_run_specs(
    root_dir=save_root,
    target_hidden1=hidden1,
    target_hidden2=hidden2,
    target_z_dim=z_dim,
    target_seed=seed,
    lr_filter=target_lr,
    epochs_filter=target_epochs,
  )
  if not runs:
    raise RuntimeError(
      f"no runs found for h1={hidden1}, h2={hidden2}, z={z_dim}, seed={seed}, lr={target_lr}, epochs={target_epochs}"
    )
  # print("selected runs:")
  # for run in runs:
  #   print(f"- beta={run.beta:g} lr={run.lr:g} epochs={run.epochs} dir={run.run_name}")

  for current_layers_to_prune in prune_layer_sets:
    curves = run_pruning_stability(
      runs=runs,
      hidden1=hidden1,
      hidden2=hidden2,
      z_dim=z_dim,
      layer_names=current_layers_to_prune,
      prune_fn=neuron_prune_layers,
      prune_percent_values=prune_percents,
    )
    save_path = plot_curves(curves, hidden1, hidden2, z_dim, seed, current_layers_to_prune)
    print(f"\nplot saved to: {save_path}")

  # input("press enter to terminate...")
