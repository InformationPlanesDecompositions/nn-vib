#!/usr/bin/env python3
import argparse, json, os, re
from dataclasses import dataclass
from datetime import datetime, timezone
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_ib import VIBNet, evaluate_epoch
from msc import CIFAR10Dataset, get_device

# model and evaluation config
input_shape = (3, 32, 32)
output_shape = 10
batch_size = 512

# pruning experiment config
prune_layer_sets = [["fc2"]]
prune_methods = ["incoming", "outgoing", "weight"]
outgoing_layer_map = {
  "fc_mu_logvar": ("fc2", "latent_linear"),
  "fc2": ("fc_decode", "linear"),
  "fc_decode": None,
}
prune_percents = [
  0.0, 0.05, 0.1,
  0.15, 0.2, 0.25,
  0.3, 0.35, 0.4,
  0.45, 0.5, 0.55,
  0.6, 0.65, 0.7,
  0.75, 0.8,
]

@dataclass(frozen=True)
class RunSpec:
  run_dir: str
  run_name: str
  hidden1: int
  hidden2: int
  decoder_hidden: int
  z_dim: int
  beta: float
  lr: float
  epochs: int
  seed: int

def format_layer_set(layer_names: list[str]) -> str: return json.dumps(layer_names)

def get_prunable_layer(model: nn.Module, layer_name: str) -> nn.Module:
  layer = dict(model.named_modules()).get(layer_name)
  if layer is None:
    raise ValueError(f"layer not found in model: {layer_name}")
  if not isinstance(layer, (nn.Linear, nn.Conv2d)):
    raise ValueError(f"layer is not nn.Linear or nn.Conv2d: {layer_name}")
  return layer

def output_unit_count(module: nn.Module) -> int:
  if isinstance(module, nn.Linear):
    return module.weight.shape[0]
  if isinstance(module, nn.Conv2d):
    return module.weight.shape[0]
  raise ValueError(f"unsupported module type: {type(module).__name__}")

def lowest_score_indices(scores: torch.Tensor, amount: float) -> torch.Tensor:
  prune_count = int(round(amount * scores.numel()))
  prune_count = max(0, min(scores.numel(), prune_count))
  if prune_count == 0:
    return torch.empty(0, dtype=torch.long, device=scores.device)
  return torch.topk(scores, k=prune_count, largest=False).indices

def weight_prune_layers(model: nn.Module, layer_names: list[str], amount: float) -> None:
  if amount <= 0:
    return

  for layer_name in layer_names:
    module = get_prunable_layer(model, layer_name)

    weight_count = module.weight.numel()
    prune_count = int(round(amount * weight_count))
    prune_count = max(0, min(weight_count, prune_count))
    if prune_count == 0:
      continue

    with torch.no_grad():
      flat_abs = module.weight.detach().abs().reshape(-1)
      prune_idx = torch.topk(flat_abs, k=prune_count, largest=False).indices
      flat_weight = module.weight.reshape(-1)
      flat_weight[prune_idx] = 0

def incoming_prune_layers(model: nn.Module, layer_names: list[str], amount: float) -> None:
  if amount <= 0:
    return

  for layer_name in layer_names:
    module = get_prunable_layer(model, layer_name)
    if isinstance(module, nn.Linear):
      scores = module.weight.detach().abs().mean(dim=1)
    else:
      scores = module.weight.detach().abs().mean(dim=(1, 2, 3))

    prune_idx = lowest_score_indices(scores, amount)
    if prune_idx.numel() == 0:
      continue

    with torch.no_grad():
      if isinstance(module, nn.Linear):
        module.weight[prune_idx, :] = 0
      else:
        module.weight[prune_idx, :, :, :] = 0

      if module.bias is not None:
        module.bias[prune_idx] = 0

def outgoing_prune_layers(model: VIBNet, layer_names: list[str], amount: float) -> None:
  if amount <= 0:
    return

  for layer_name in layer_names:
    module = get_prunable_layer(model, layer_name)
    next_layer_spec = outgoing_layer_map.get(layer_name)
    if layer_name not in outgoing_layer_map:
      raise ValueError(f"no outgoing layer mapping configured for: {layer_name}")
    if next_layer_spec is None:
      raise ValueError(f"layer has no outgoing layer to prune against: {layer_name}")

    next_layer_name, edge_type = next_layer_spec
    next_layer = get_prunable_layer(model, next_layer_name)

    if edge_type == "linear":
      if not isinstance(next_layer, nn.Linear):
        raise ValueError(f"expected linear next layer for {next_layer_name}")
      if next_layer.weight.shape[1] != output_unit_count(module):
        raise ValueError(
          f"shape mismatch between {layer_name} outputs and {next_layer_name} inputs: "
          f"{output_unit_count(module)} != {next_layer.weight.shape[1]}"
        )
      scores = next_layer.weight.detach().abs().mean(dim=0)
    elif edge_type == "latent_linear":
      if layer_name != "fc_mu_logvar" or not isinstance(next_layer, nn.Linear):
        raise ValueError(f"unsupported latent-linear mapping: {layer_name} -> {next_layer_name}")
      scores = next_layer.weight.detach().abs().mean(dim=0)
    else:
      raise ValueError(f"unsupported outgoing edge type: {edge_type}")

    prune_idx = lowest_score_indices(scores, amount)
    if prune_idx.numel() == 0:
      continue

    with torch.no_grad():
      if edge_type in ("linear", "latent_linear"):
        next_layer.weight[:, prune_idx] = 0
      else:
        raise ValueError(f"unsupported outgoing edge type: {edge_type}")

def prune_layers(model: VIBNet, layer_names: list[str], amount: float, prune_method: str) -> None:
  if prune_method == "weight":
    weight_prune_layers(model, layer_names, amount)
    return
  if prune_method == "incoming":
    incoming_prune_layers(model, layer_names, amount)
    return
  if prune_method == "outgoing":
    outgoing_prune_layers(model, layer_names, amount)
    return
  raise ValueError(f"unknown prune method: {prune_method}")

def parse_all_run_specs(root_dir: str) -> list[RunSpec]:
  pattern = re.compile(r"^vib_cnn_(\d+)_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$")
  runs = []
  for run_name in os.listdir(root_dir):
    run_dir = os.path.join(root_dir, run_name)
    if not os.path.isdir(run_dir):
      continue

    match = pattern.match(run_name)
    if not match:
      continue

    h1_s, h2_s, decoder_hidden_s, z_s, beta_s, lr_s, epochs_s, seed_s = match.groups()
    runs.append(
      RunSpec(
        run_dir=run_dir,
        run_name=run_name,
        hidden1=int(h1_s),
        hidden2=int(h2_s),
        decoder_hidden=int(decoder_hidden_s),
        z_dim=int(z_s),
        beta=float(beta_s),
        lr=float(lr_s),
        epochs=int(epochs_s),
        seed=int(seed_s),
      )
    )
  return sorted(
    runs,
    key=lambda run: (run.hidden1, run.hidden2, run.decoder_hidden, run.z_dim, run.seed, run.lr, run.epochs, run.beta),
  )

def load_state_dict_from_run(run_dir: str) -> dict[str, torch.Tensor]:
  pth_files = sorted([name for name in os.listdir(run_dir) if name.endswith(".pth")])
  if not pth_files:
    raise FileNotFoundError(f"no .pth file found in {run_dir}")
  return torch.load(os.path.join(run_dir, pth_files[0]), map_location="cpu")

def config_key(run: RunSpec) -> tuple[int, int, int, int, int, float, int]:
  return (run.hidden1, run.hidden2, run.decoder_hidden, run.z_dim, run.seed, run.lr, run.epochs)

def group_runs_by_config(runs: list[RunSpec]) -> list[tuple[tuple[int, int, int, int, int, float, int], list[RunSpec]]]:
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
  prune_method: str,
  test_loader: DataLoader,
  device: torch.device,
) -> dict[str, object]:
  print(f"  beta={run.beta:g} run={run.run_name}")
  state_dict = load_state_dict_from_run(run.run_dir)
  losses, accs = [], []

  for prune_percent in prune_percents:
    model = VIBNet(run.z_dim, input_shape, run.hidden1, run.hidden2, run.decoder_hidden, output_shape).to(device)
    model.load_state_dict(state_dict)

    prune_layers(model, layer_names, prune_percent, prune_method)
    loss, acc = evaluate_epoch(model, test_loader, beta=run.beta)

    losses.append(float(loss))
    accs.append(float(acc))
    print(f"    prune={prune_percent * 100:>5.1f}% loss={loss:.6f} acc={acc:.2f}")

  return {
    "beta": run.beta,
    "run_name": run.run_name,
    "run_dir": run.run_dir,
    "prune_percents": prune_percents,
    "losses": losses,
    "accuracies": accs,
  }

def build_report(save_root: str, data_dir: str, prune_method: str) -> dict[str, object]:
  runs = parse_all_run_specs(save_root)
  if not runs:
    raise RuntimeError(f"no vib_cnn_* runs found in {save_root}")

  device = get_device()
  test_loader = DataLoader(
    CIFAR10Dataset(data_dir, train=False, transform=CIFAR10Dataset.test_transform(data_dir)),
    batch_size=batch_size,
    shuffle=False,
  )

  configs = []
  grouped_runs = group_runs_by_config(runs)
  for config_idx, (_, config_runs) in enumerate(grouped_runs, start=1):
    first_run = config_runs[0]
    print(
      f"[{config_idx}/{len(grouped_runs)}] h1={first_run.hidden1} h2={first_run.hidden2} "
      f"decoder={first_run.decoder_hidden} z={first_run.z_dim} seed={first_run.seed} "
      f"lr={first_run.lr:g} epochs={first_run.epochs}"
    )

    layer_results = []
    for layer_names in prune_layer_sets:
      print(f"  layers={format_layer_set(layer_names)}")
      curves = []
      for run in config_runs:
        curves.append(evaluate_pruning_curve(run, layer_names, prune_method, test_loader, device))
      layer_results.append({"layer_names": layer_names, "curves": curves})

    configs.append(
      {
        "hidden1": first_run.hidden1,
        "hidden2": first_run.hidden2,
        "decoder_hidden": first_run.decoder_hidden,
        "z_dim": first_run.z_dim,
        "seed": first_run.seed,
        "lr": first_run.lr,
        "epochs": first_run.epochs,
        "layer_results": layer_results,
      }
    )

  return {
    "save_root": os.path.abspath(save_root),
    "data_dir": os.path.abspath(data_dir),
    "prune_method": prune_method,
    "prune_percents": prune_percents,
    "layer_sets": prune_layer_sets,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "config_count": len(configs),
    "run_count": len(runs),
    "configs": configs,
  }

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="inspect cnn pruning stability across an entire save root")
  parser.add_argument("--save_root", type=str, required=True, help="directory containing saved model runs")
  parser.add_argument("--data_dir", type=str, default="data/CIFAR-10/", help="cifar-10 dataset path")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  if not os.path.isdir(args.save_root):
    raise RuntimeError(f"save_root does not exist or is not a directory: {args.save_root}")
  if not os.path.isdir(args.data_dir):
    raise RuntimeError(f"data_dir does not exist or is not a directory: {args.data_dir}")

  for prune_method in prune_methods:
    print(f"\nprune_method={prune_method}")
    report = build_report(args.save_root, args.data_dir, prune_method)
    json_path = os.path.join(args.save_root, f"cnn_pruning_report_{prune_method}.json")
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"\njson saved to: {json_path}")
