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

plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 13

default_plots_dir = "../Plots"
beta_zero = 0.0
metric_choices = ["loss", "acc"]

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

def loss_robustness_curve(reference_values: list[float], values: list[float]) -> np.ndarray:
  eps = 1e-8
  reference = np.asarray(reference_values, dtype=np.float64)
  arr = np.asarray(values, dtype=np.float64)
  return reference / np.clip(arr, eps, None)

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

def metric_ylabel(metric: str) -> str:
  if metric == "loss":
    return "Robustness" #(L(i,0,p) / L(i,beta,p))
  if metric == "acc":
    return "Accuracy robustness" #(Acc(i,beta,p) / Acc(i,0,p))
  raise ValueError(f"unknown metric: {metric}")

def filtered_beta_items(beta_samples: dict[float, list[np.ndarray]]) -> list[tuple[float, list[np.ndarray]]]:
  return [(beta, beta_samples[beta]) for beta in sorted(beta_samples)]

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
  metric: str,
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
        if metric == "loss":
          raise ValueError(
            f"missing beta=0 curve for config {config_key} and method {method_key}; cannot compute L(i,0,p)/L(i,beta,p)"
          )
        if metric == "acc":
          raise ValueError(
            f"missing beta=0 curve for config {config_key} and method {method_key}; cannot compute Acc(i,beta,p)/Acc(i,0,p)"
          )
        raise ValueError(f"unknown metric: {metric}")

      for curve in layer_result["curves"]:
        prune_percents = curve["prune_percents"]
        existing = prune_percent_map.get(method_key)
        if existing is None:
          prune_percent_map[method_key] = prune_percents
        elif existing != prune_percents:
          raise ValueError(f"inconsistent prune percents for method {method_key}")

        if metric == "loss":
          normalized = loss_robustness_curve(beta_zero_curve["losses"], curve["losses"])
        elif metric == "acc":
          normalized = accuracy_robustness_curve(beta_zero_curve["accuracies"], curve["accuracies"])
        else:
          raise ValueError(f"unknown metric: {metric}")

        beta = float(curve["beta"])
        panel_samples[method_key][config_key][beta].append(normalized)

  return panel_samples, prune_percent_map

def plot_page(
  method_key: tuple[str, ...],
  config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]],
  prune_percents: list[float],
  prune_method: str,
  metric: str,
  output_dir: str,
  page_index: int,
  page_count: int,
) -> str:
  fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
  axes_list = list(axes.flat)
  xs = np.asarray(prune_percents, dtype=np.float64)

  for ax, (config_key, beta_samples) in zip(axes_list, config_items):
    for beta, curves in filtered_beta_items(beta_samples):
      mean, stderr = summarize_curves(curves)
      ax.plot(xs, mean, marker="o", linewidth=1.8, label=f"beta={beta:g}")
      ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_title(model_config_label(config_key))
    ax.set_xlabel("Pruning fraction")
    ax.set_ylabel(metric_ylabel(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.4, fontsize=9)

  for ax in axes_list[len(config_items) :]:
    ax.axis("off")

  #fig.suptitle(
  #  f"MLP pruning | method={prune_method} | pruned: {', '.join(method_key)} | page {page_index}/{page_count} | {metric}",
  #  fontsize=16,
  #)

  page_suffix = "" if page_count == 1 else f"_part_{page_index}"
  save_name = f"mlp_pruning_{prune_method}_{layer_key(list(method_key))}{page_suffix}_{metric}.png"
  save_path = os.path.join(output_dir, save_name)
  fig.savefig(save_path, dpi=250, bbox_inches="tight")
  plt.close(fig)
  return save_path

def plot_aggregate_page(
  method_key: tuple[str, ...],
  config_items: list[tuple[tuple[object, ...], dict[float, list[np.ndarray]]]],
  prune_percents: list[float],
  prune_method: str,
  metric: str,
  output_dir: str,
) -> str:
  fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
  xs = np.asarray(prune_percents, dtype=np.float64)
  aggregate_samples = aggregate_beta_samples(config_items)

  for beta, curves in filtered_beta_items(aggregate_samples):
    mean, stderr = summarize_curves(curves)
    ax.plot(xs, mean, marker="o", linewidth=2.0, label=f"beta={beta:g}")
    ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)

  ax.set_xlabel("Pruning fraction")
  ax.set_ylabel(metric_ylabel(metric))
  #ax.set_title(f"Average over model sizes | pruned: {', '.join(method_key)}")
  ax.grid(True, alpha=0.3)
  ax.legend(framealpha=0.4)

  #fig.suptitle(
  #  f"MLP pruning aggregate | method={prune_method} | {metric}",
  #  fontsize=16,
  #)

  save_name = f"mlp_pruning_{prune_method}_{layer_key(list(method_key))}_aggregate_{metric}.png"
  save_path = os.path.join(output_dir, save_name)
  fig.savefig(save_path, dpi=250, bbox_inches="tight")
  plt.close(fig)
  return save_path

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="plot aggregated mlp pruning robustness from a json report")
  parser.add_argument("--input_json", type=str, required=True, help="json report produced by inspect_mlp_ib.py")
  parser.add_argument("--plots_dir", type=str, default=default_plots_dir, help="directory to save plots into")
  parser.add_argument("--metric", choices=metric_choices, default="loss", help="plot robustness from losses or accuracies")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  with open(args.input_json, "r", encoding="utf-8") as f:
    report = json.load(f)

  prune_method = report["prune_method"]
  plots_dir = args.plots_dir
  os.makedirs(plots_dir, exist_ok=True)

  panel_samples, prune_percent_map = build_samples(report, args.metric)

  for method_key in sorted(panel_samples):
    sorted_config_items = sorted(panel_samples[method_key].items(), key=lambda item: item[0])
    aggregate_path = plot_aggregate_page(
      method_key,
      sorted_config_items,
      prune_percent_map[method_key],
      prune_method,
      args.metric,
      plots_dir,
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
        args.metric,
        plots_dir,
        page_idx + 1,
        page_count,
      )
      print(f"saved figure: {save_path}")
