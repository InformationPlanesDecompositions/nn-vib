#!/usr/bin/env python3
import argparse
import json
import math
import os

import matplotlib

if "MPLBACKEND" not in os.environ:
  matplotlib.use("Agg")

import matplotlib.pyplot as plt

default_plots_dir = "plots"
subplot_dir_name = "subplots"


def safe_float_str(value: float) -> str:
  return format(value, "g").replace("-", "m").replace(".", "p")


def layer_key(layer_names: list[str]) -> str:
  return "_".join(layer_names)


def config_sort_key(config: dict[str, object]) -> tuple[object, ...]:
  return (
    config["hidden1"],
    config["hidden2"],
    config["z_dim"],
    config["seed"],
    config["lr"],
    config["epochs"],
  )


def plot_layer_result(ax, config: dict[str, object], layer_result: dict[str, object], prune_axis: str) -> None:
  for curve in sorted(layer_result["curves"], key=lambda item: item["beta"]):
    xs = [value * 100 for value in curve["prune_percents"]]
    ax.plot(xs, curve["losses"], marker="o", linewidth=1.8, label=f"beta={curve['beta']:g}")

  layer_part = ", ".join(layer_result["layer_names"])
  title = (
    f"({config['hidden1']}, {config['hidden2']}, {config['z_dim']}) seed={config['seed']}\n"
    f"lr={config['lr']:g} epochs={config['epochs']} | axis={prune_axis}\n"
    f"pruned: {layer_part}"
  )
  ax.set_title(title, fontsize=9)
  ax.set_xlabel("percent pruned")
  ax.set_ylabel("test loss")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=7)


def save_subplot(
  config: dict[str, object],
  layer_result: dict[str, object],
  prune_axis: str,
  subplot_dir: str,
) -> str:
  fig, ax = plt.subplots(figsize=(10, 6))
  plot_layer_result(ax, config, layer_result, prune_axis)
  fig.tight_layout()
  save_name = (
    f"mlp_prune_{prune_axis}_h1_{config['hidden1']}_h2_{config['hidden2']}_z_{config['z_dim']}"
    f"_seed_{config['seed']}_lr_{safe_float_str(config['lr'])}_epochs_{config['epochs']}"
    f"_{layer_key(layer_result['layer_names'])}.png"
  )
  save_path = os.path.join(subplot_dir, save_name)
  fig.savefig(save_path, dpi=250, bbox_inches="tight")
  plt.close(fig)
  return save_path


def save_overview(
  panels: list[tuple[dict[str, object], dict[str, object]]],
  prune_axis: str,
  source_json: str,
  plots_dir: str,
) -> str:
  panel_count = len(panels)
  cols = 3 if panel_count <= 12 else 4
  rows = math.ceil(panel_count / cols)
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), constrained_layout=True)
  if hasattr(axes, "flat"):
    axes_list = list(axes.flat)
  else:
    axes_list = [axes]

  for ax, (config, layer_result) in zip(axes_list, panels):
    plot_layer_result(ax, config, layer_result, prune_axis)

  for ax in axes_list[panel_count:]:
    ax.axis("off")

  fig.suptitle(
    f"MLP pruning overview | axis={prune_axis} | report={os.path.basename(source_json)}",
    fontsize=16,
  )
  overview_name = f"mlp_pruning_overview_{prune_axis}.png"
  overview_path = os.path.join(plots_dir, overview_name)
  fig.savefig(overview_path, dpi=250, bbox_inches="tight")
  plt.close(fig)
  return overview_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="plot mlp pruning results from a json report")
  parser.add_argument("--input_json", type=str, required=True, help="json report produced by inspect_mlp_ib.py")
  parser.add_argument("--plots_dir", type=str, default=default_plots_dir, help="directory to save plots into")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  with open(args.input_json, "r", encoding="utf-8") as f:
    report = json.load(f)

  prune_axis = report["prune_axis"]
  plots_dir = args.plots_dir
  subplot_dir = os.path.join(plots_dir, subplot_dir_name)
  os.makedirs(subplot_dir, exist_ok=True)

  sorted_configs = sorted(report["configs"], key=config_sort_key)
  panels = []
  for config in sorted_configs:
    for layer_result in config["layer_results"]:
      panels.append((config, layer_result))
  for config, layer_result in panels:
    save_path = save_subplot(config, layer_result, prune_axis, subplot_dir)
    print(f"saved subplot: {save_path}")

  overview_path = save_overview(panels, prune_axis, args.input_json, plots_dir)
  print(f"saved overview: {overview_path}")
