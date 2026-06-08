#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict
import matplotlib

if "MPLBACKEND" not in os.environ: matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

default_plots_dir = "../Plots"
beta_zero = 0.0
beta_label_choices = ["table", "colorbar"]

def safe_float_str(value: float) -> str: return format(value, "g").replace("-", "m").replace(".", "p")

def layer_key(layer_names: list[str]) -> str: return "_".join(layer_names)

def model_config_key(config: dict[str, object]) -> tuple[object, ...]:
  return (
    config["hidden1"],
    config["hidden2"],
    config["z_dim"],
    config["lr"],
    config["epochs"],
  )

def model_config_label(config_key: tuple[object, ...]) -> str:
  hidden1, hidden2, z_dim, lr, epochs = config_key
  return f"h1={hidden1} h2={hidden2} z={z_dim}\nlr={float(lr):g} epochs={epochs}"

def accuracy_robustness_curve(reference_values: list[float], values: list[float]) -> np.ndarray:
  eps = 1e-8
  reference = np.asarray(reference_values, dtype=np.float64)
  arr = np.asarray(values, dtype=np.float64)
  return arr / np.clip(reference, eps, None)

def summarize_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  stacked = np.stack(curves, axis=0)
  mean = stacked.mean(axis=0)
  if stacked.shape[0] == 1:
    return mean, np.zeros_like(mean)
  stderr = stacked.std(axis=0, ddof=1) / math.sqrt(stacked.shape[0])
  return mean, stderr

def filtered_beta_items(beta_samples: dict[float, list[np.ndarray]]) -> list[tuple[float, list[np.ndarray]]]:
  return [(beta, beta_samples[beta]) for beta in sorted(beta_samples)]

def beta_bounds(config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]]) -> tuple[float, float]:
  betas = [beta for _, beta_samples in config_items for beta in beta_samples]
  return min(betas), max(betas)

def add_beta_colorbar(fig, axes, beta_min: float, beta_max: float) -> tuple[object, object]:
  cmap = plt.get_cmap("RdYlGn_r")
  norm = matplotlib.colors.Normalize(vmin=beta_min, vmax=beta_max)
  scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  fig.colorbar(scalar_map, ax=axes, label="beta", ticks=[beta_min, beta_max])
  return cmap, norm

def aggregate_beta_samples(
  config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]],
) -> dict[float, list[np.ndarray]]:
  aggregate_samples: dict[float, list[np.ndarray]] = defaultdict(list)
  for _, beta_samples in config_items:
    for beta, curves in beta_samples.items():
      config_mean, _ = summarize_curves(curves)
      aggregate_samples[beta].append(config_mean)
  return aggregate_samples

def build_samples(
  report: dict[str, object],
) -> tuple[
  dict[tuple[str, ...], dict[tuple[object, ...], dict[float, list[np.ndarray]]]],
  dict[tuple[str, ...], list[float]],
]:
  panel_samples: dict[tuple[str, ...], dict[tuple[object, ...], dict[float, list[np.ndarray]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
  )
  prune_percent_map: dict[tuple[str, ...], list[float]] = {}

  for config in report["configs"]:
    config_key = model_config_key(config)
    for layer_result in config["layer_results"]:
      method_key = tuple(layer_result["layer_names"])
      beta_zero_curve = next(
        (curve for curve in layer_result["curves"] if math.isclose(float(curve["beta"]), beta_zero)),
        None,
      )
      if beta_zero_curve is None:
        raise ValueError(
          f"missing beta=0 curve for config {config_key} and method {method_key}; cannot compute Acc(i,beta,p)/Acc(i,0,p)"
        )

      for curve in layer_result["curves"]:
        prune_percents = curve["prune_percents"]
        existing = prune_percent_map.get(method_key)
        if existing is None:
          prune_percent_map[method_key] = prune_percents
        elif existing != prune_percents:
          raise ValueError(f"inconsistent prune percents for method {method_key}")

        normalized = accuracy_robustness_curve(beta_zero_curve["accuracies"], curve["accuracies"])

        beta = float(curve["beta"])
        panel_samples[method_key][config_key][beta].append(normalized)

  return panel_samples, prune_percent_map

def plot_page(
  method_key: tuple[str, ...],
  config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]],
  prune_percents: list[float],
  prune_method: str,
  output_dir: str,
  page_index: int,
  page_count: int,
  beta_labels: str,
) -> str:
  fig, axes = plt.subplots(2, 3)
  axes_list = list(axes.flat)
  xs = np.asarray(prune_percents, dtype=np.float64)
  cmap, norm = None, None
  if beta_labels == "colorbar":
    beta_min, beta_max = beta_bounds(config_items)
    cmap, norm = add_beta_colorbar(fig, axes_list[:len(config_items)], beta_min, beta_max)

  for ax, (config_key, beta_samples) in zip(axes_list, config_items):
    for beta, curves in filtered_beta_items(beta_samples):
      mean, stderr = summarize_curves(curves)
      if beta_labels == "colorbar":
        ax.plot(xs, mean, color=cmap(norm(beta)))
      else:
        ax.plot(xs, mean, label=f"beta={beta:g}")
    ax.set_title(model_config_label(config_key))
    ax.set_xlabel("Pruning fraction")
    ax.set_ylabel("Accuracy robustness")
    if beta_labels == "table":
      ax.legend()

  for ax in axes_list[len(config_items) :]:
    ax.axis("off")

  page_suffix = "" if page_count == 1 else f"_part_{page_index}"
  save_name = f"mlp_pruning_{prune_method}_{layer_key(list(method_key))}{page_suffix}_acc.png"
  save_path = os.path.join(output_dir, save_name)
  fig.savefig(save_path)
  plt.close(fig)
  return save_path

def plot_aggregate_page(
  method_key: tuple[str, ...],
  config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]],
  prune_percents: list[float],
  prune_method: str,
  output_dir: str,
  beta_labels: str,
) -> str:
  fig, ax = plt.subplots()
  xs = np.asarray(prune_percents, dtype=np.float64)
  aggregate_samples = aggregate_beta_samples(config_items)
  cmap, norm = None, None
  if beta_labels == "colorbar":
    beta_min, beta_max = min(aggregate_samples), max(aggregate_samples)
    cmap, norm = add_beta_colorbar(fig, ax, beta_min, beta_max)

  for beta, curves in filtered_beta_items(aggregate_samples):
    mean, stderr = summarize_curves(curves)
    if beta_labels == "colorbar":
      ax.plot(xs, mean, color=cmap(norm(beta)))
    else:
      ax.plot(xs, mean, label=f"beta={beta:g}")

  ax.set_xlabel("Pruning fraction")
  ax.set_ylabel("Accuracy robustness")
  if beta_labels == "table":
    ax.legend()

  save_name = f"mlp_pruning_{prune_method}_{layer_key(list(method_key))}_aggregate_acc.png"
  save_path = os.path.join(output_dir, save_name)
  fig.savefig(save_path)
  plt.close(fig)
  return save_path

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="plot aggregated mlp pruning robustness from a json report")
  parser.add_argument("--input_json", type=str, required=True, help="json report produced by inspect_mlp_ib.py")
  parser.add_argument("--plots_dir", type=str, default=default_plots_dir, help="directory to save plots into")
  parser.add_argument("--beta_labels", choices=beta_label_choices, default="table", help="label beta lines with a table or colorbar")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  with open(args.input_json, "r", encoding="utf-8") as f:
    report = json.load(f)

  prune_method = report["prune_method"]
  plots_dir = args.plots_dir
  os.makedirs(plots_dir, exist_ok=True)

  panel_samples, prune_percent_map = build_samples(report)

  for method_key in sorted(panel_samples):
    sorted_config_items = sorted(panel_samples[method_key].items(), key=lambda item: item[0])
    aggregate_path = plot_aggregate_page(
      method_key,
      sorted_config_items,
      prune_percent_map[method_key],
      prune_method,
      plots_dir,
      args.beta_labels,
    )
    print(f"saved figure: {aggregate_path}")
    page_size = 6
    page_count = math.ceil(len(sorted_config_items) / page_size)
    for page_idx in range(page_count):
      start = page_idx * page_size
      end = start + page_size
      save_path = plot_page(
        method_key,
        sorted_config_items[start:end],
        prune_percent_map[method_key],
        prune_method,
        plots_dir,
        page_idx + 1,
        page_count,
        args.beta_labels,
      )
      print(f"saved figure: {save_path}")
