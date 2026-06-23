#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mlp_ib import VIBMLP, evaluate_epoch
from msc import FashionMnistIdxDataset, get_device


prune_percents = [i / 100 for i in range(0, 81, 5)]
mlp2 = (256, 64, 10)
layers = ["fc_mu_logvar", "fc2"]
axis_label_fontsize = 20
tick_fontsize = 15
legend_fontsize = 13


def find_runs(save_root):
    pattern = re.compile(
        r"^vib_mlp_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$"
    )
    runs = []
    for name in os.listdir(save_root):
        match = pattern.match(name)
        if not match:
            continue
        h1, h2, z, beta, lr, epochs, seed = match.groups()
        if (int(h1), int(h2), int(z)) == mlp2:
            runs.append((os.path.join(save_root, name), float(beta), int(seed)))
    return sorted(runs, key=lambda run: (run[1], run[2]))


def load_state_dict(run_dir):
    pth_files = sorted(name for name in os.listdir(run_dir) if name.endswith(".pth"))
    if not pth_files:
        raise FileNotFoundError(f"no .pth file found in {run_dir}")
    return torch.load(os.path.join(run_dir, pth_files[0]), map_location="cpu")


def prune_weight_magnitude(model, layer_name, amount):
    if amount <= 0:
        return
    layer = dict(model.named_modules())[layer_name]
    prune_count = int(round(amount * layer.weight.numel()))
    if prune_count == 0:
        return
    with torch.no_grad():
        flat_abs = layer.weight.detach().abs().reshape(-1)
        prune_idx = torch.topk(flat_abs, k=prune_count, largest=False).indices
        layer.weight.reshape(-1)[prune_idx] = 0


def curve_for_run(run_dir, beta, loader, device):
    state_dict = load_state_dict(run_dir)
    accs = []
    for amount in prune_percents:
        model = VIBMLP(10, 784, 256, 64, 10).to(device)
        model.load_state_dict(state_dict)
        for layer_name in layers:
            prune_weight_magnitude(model, layer_name, amount)
        _, acc = evaluate_epoch(model, loader, beta)
        accs.append(acc)
    return np.asarray(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--output", default="mlp2_weight_pruning_accuracy.pdf")
    args = parser.parse_args()

    runs = find_runs(args.save_root)
    if not runs:
        raise RuntimeError(f"no MLP2 runs found in {args.save_root}")

    device = get_device()
    loader = DataLoader(
        FashionMnistIdxDataset("data/mnist_fashion/", train=False),
        batch_size=128,
        shuffle=False,
    )

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    xs = np.asarray(prune_percents)

    by_beta = defaultdict(list)
    for run_dir, beta, seed in runs:
        print(f"beta={beta:g} seed={seed}")
        by_beta[beta].append(curve_for_run(run_dir, beta, loader, device))

    for beta in sorted(by_beta):
        mean = np.stack(by_beta[beta]).mean(axis=0)
        ax.plot(xs, mean, label=f"beta={beta:g}")

    ax.set_xlabel("Pruning Fraction", fontsize=axis_label_fontsize)
    ax.set_ylabel("Accuracy (%)", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)

    fig.savefig(args.output)
    print(f"saved plot: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
