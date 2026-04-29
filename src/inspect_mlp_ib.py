#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlp_ib import VIBNet
from msc import FashionMnistIdxDataset, evaluate_epoch, get_device

# model and evaluation config
input_shape = 784
output_shape = 10
batch_size = 128

# pruning experiment config
layers_to_prune = ["fc_mu", "fc_logvar", "fc2"]
layers_to_prune_fc2_only = ["fc2"]
prune_layer_sets = [layers_to_prune, layers_to_prune_fc2_only]
prune_percents = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


@dataclass(frozen=True)
class RunSpec:
  run_dir: str
  run_name: str
  hidden1: int
  hidden2: int
  z_dim: int
  beta: float
  lr: float
  epochs: int
  seed: int


def indices_above_avg_abs_threshold(
  model: nn.Module,
  layer_name: str,
  count: int,
  axis: str = "row",
  largest: bool = True,
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
  avg_abs_by_idx.sort(key=lambda x: x[1], reverse=largest)
  selected_pairs = avg_abs_by_idx[:count]
  return {idx: avg for idx, avg in selected_pairs}


def inspect_top_neurons(
  model: nn.Module,
  layer_names: list[str],
  count: int | dict[str, int],
  axis: str = "row",
  largest: bool = True,
) -> dict[str, list[int]]:
  top_neurons = {}
  for layer_name in layer_names:
    layer_count = count[layer_name] if isinstance(count, dict) else count
    indices = indices_above_avg_abs_threshold(model, layer_name, layer_count, axis=axis, largest=largest)
    top_neurons[layer_name] = list(indices.keys())
  return top_neurons


def neuron_prune_layers(model: nn.Module, layer_names: list[str], amount: float, axis: str = "row") -> None:
  if amount <= 0:
    return
  if axis not in {"row", "col"}:
    raise ValueError("axis must be 'row' or 'col'")
  modules = dict(model.named_modules())
  prune_counts = {}
  for layer_name in layer_names:
    module = modules.get(layer_name)
    if module is None:
      raise ValueError(f"layer not found in model: {layer_name}")
    if not isinstance(module, nn.Linear):
      raise ValueError(f"layer is not nn.Linear: {layer_name}")
    axis_count = module.weight.shape[0] if axis == "row" else module.weight.shape[1]
    prune_count = int(round(amount * axis_count))
    prune_counts[layer_name] = max(0, min(axis_count, prune_count))

  indices_to_prune = inspect_top_neurons(model, layer_names, prune_counts, axis=axis, largest=False)

  for layer_name, prune_indices in indices_to_prune.items():
    if not prune_indices:
      continue
    module = modules[layer_name]
    with torch.no_grad():
      prune_idx = torch.tensor(prune_indices, device=module.weight.device)
      if axis == "row":
        module.weight[prune_idx, :] = 0
        if module.bias is not None:
          module.bias[prune_idx] = 0
      else:
        module.weight[:, prune_idx] = 0


def parse_all_run_specs(root_dir: str) -> list[RunSpec]:
  pattern = re.compile(r"^vib_mlp_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$")
  runs = []
  for run_name in os.listdir(root_dir):
    run_dir = os.path.join(root_dir, run_name)
    if not os.path.isdir(run_dir):
      continue
    match = pattern.match(run_name)
    if not match:
      continue
    h1_s, h2_s, z_s, beta_s, lr_s, epochs_s, seed_s = match.groups()
    runs.append(
      RunSpec(
        run_dir=run_dir,
        run_name=run_name,
        hidden1=int(h1_s),
        hidden2=int(h2_s),
        z_dim=int(z_s),
        beta=float(beta_s),
        lr=float(lr_s),
        epochs=int(epochs_s),
        seed=int(seed_s),
      )
    )
  return sorted(runs, key=lambda run: (run.hidden1, run.hidden2, run.z_dim, run.seed, run.lr, run.epochs, run.beta))


def load_state_dict_from_run(run_dir: str) -> dict[str, torch.Tensor]:
  pth_files = sorted([name for name in os.listdir(run_dir) if name.endswith(".pth")])
  if not pth_files:
    raise FileNotFoundError(f"no .pth file found in {run_dir}")
  return torch.load(os.path.join(run_dir, pth_files[0]), map_location="cpu")


def config_key(run: RunSpec) -> tuple[int, int, int, int, float, int]:
  return (run.hidden1, run.hidden2, run.z_dim, run.seed, run.lr, run.epochs)


def group_runs_by_config(runs: list[RunSpec]) -> list[tuple[tuple[int, int, int, int, float, int], list[RunSpec]]]:
  grouped = {}
  for run in runs:
    key = config_key(run)
    grouped.setdefault(key, []).append(run)
  items = []
  for key, config_runs in grouped.items():
    items.append((key, sorted(config_runs, key=lambda run: run.beta)))
  items.sort(key=lambda item: item[0])
  return items


def evaluate_pruning_curve(
  run: RunSpec,
  layer_names: list[str],
  prune_axis: str,
  test_loader: DataLoader,
  device: torch.device,
) -> dict[str, object]:
  print(f"  beta={run.beta:g} run={run.run_name}")
  state_dict = load_state_dict_from_run(run.run_dir)
  losses = []
  accuracies = []
  for prune_percent in prune_percents:
    model = VIBNet(run.z_dim, input_shape, run.hidden1, run.hidden2, output_shape).to(device)
    model.load_state_dict(state_dict)
    neuron_prune_layers(model, layer_names, prune_percent, axis=prune_axis)
    loss, _, _, acc = evaluate_epoch(model, test_loader, device, beta=run.beta)
    losses.append(float(loss))
    accuracies.append(float(acc))
    print(f"    prune={prune_percent * 100:>5.1f}% loss={loss:.6f} acc={acc:.2f}")
  return {
    "beta": run.beta,
    "run_name": run.run_name,
    "run_dir": run.run_dir,
    "prune_percents": prune_percents,
    "losses": losses,
    "accuracies": accuracies,
  }


def build_report(save_root: str, prune_axis: str) -> dict[str, object]:
  runs = parse_all_run_specs(save_root)
  if not runs:
    raise RuntimeError(f"no vib_mlp_* runs found in {save_root}")

  device = get_device()
  test_loader = DataLoader(
    FashionMnistIdxDataset("data/mnist_fashion/", train=False),
    batch_size=batch_size,
    shuffle=False,
  )

  configs = []
  grouped_runs = group_runs_by_config(runs)
  for config_idx, (_, config_runs) in enumerate(grouped_runs, start=1):
    first_run = config_runs[0]
    print(
      f"[{config_idx}/{len(grouped_runs)}] h1={first_run.hidden1} h2={first_run.hidden2} "
      f"z={first_run.z_dim} seed={first_run.seed} lr={first_run.lr:g} epochs={first_run.epochs}"
    )
    layer_results = []
    for layer_names in prune_layer_sets:
      print(f"  layers={', '.join(layer_names)}")
      curves = []
      for run in config_runs:
        curves.append(evaluate_pruning_curve(run, layer_names, prune_axis, test_loader, device))
      layer_results.append({"layer_names": layer_names, "curves": curves})
    configs.append(
      {
        "hidden1": first_run.hidden1,
        "hidden2": first_run.hidden2,
        "z_dim": first_run.z_dim,
        "seed": first_run.seed,
        "lr": first_run.lr,
        "epochs": first_run.epochs,
        "layer_results": layer_results,
      }
    )

  return {
    "save_root": os.path.abspath(save_root),
    "prune_axis": prune_axis,
    "prune_percents": prune_percents,
    "layer_sets": prune_layer_sets,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "config_count": len(configs),
    "run_count": len(runs),
    "configs": configs,
  }


def output_json_path(save_root: str, prune_axis: str) -> str:
  return os.path.join(save_root, f"mlp_pruning_report_{prune_axis}.json")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="inspect mlp pruning stability across an entire save root")
  parser.add_argument("--save_root", type=str, required=True, help="directory containing saved model runs")
  parser.add_argument("--prune_axis", choices=["row", "col"], default="row", help="prune ranked weight rows or columns")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  if not os.path.isdir(args.save_root):
    raise RuntimeError(f"save_root does not exist or is not a directory: {args.save_root}")
  report = build_report(args.save_root, args.prune_axis)
  json_path = output_json_path(args.save_root, args.prune_axis)
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  print(f"\njson saved to: {json_path}")
