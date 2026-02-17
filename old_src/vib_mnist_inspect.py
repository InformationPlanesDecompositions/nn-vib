#!/usr/bin/env python3

import copy
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from vib_mnist_train import VIBNet, evaluate_epoch, FashionMnistIdxDataset
from msc import get_device, load_weights, weights_location

torch.manual_seed(42)
device = get_device()
print(f"using device: {device}")

block_plt = False
batch_size = 100

train_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=True)
test_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=False)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

betas = [(0.4, 0.0001), (0.3, 0.0001), (0.2, 0.0001), (0.1, 0.0005), (0.05, 0.0005), (0.01, 0.001), (0.005, 0.0005), (0.001, 0.0005), (0.0, 0.0001)]
z_dim, h1, h2, o_shape = 125, 500, 300, 10

layer_names = [
    "fc1",
    "fc_mu",
    "fc_logvar",
    "fc2",
    "fc_decode",
]

# ------------------------------------------------------------------------------

#prune_percs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
prune_percs = [i/30 for i in range(6, 30)]
print(f"prune percs: {prune_percs}")

def prune_model_per_layer(weights, axes, beta):
    model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
    model.load_state_dict(weights)

    for idx, layer_name in enumerate(layer_names):
        pruned_acc_list = []
        for prune_perc in prune_percs:
            pruned_model = copy.deepcopy(model)

            if isinstance(layer_name, tuple):
                for sub_layer in layer_name:
                    module = dict(pruned_model.named_modules())[sub_layer]
                    prune.l1_unstructured(module, name="weight", amount=prune_perc)
            else:
                module = dict(pruned_model.named_modules())[layer_name]
                prune.l1_unstructured(module, name="weight", amount=prune_perc)

            test_loss, ce, kl, test_acc = evaluate_epoch(pruned_model, test_loader, device, beta=beta)
            pruned_acc_list.append(test_loss)

        layer_label = ', '.join(layer_name) if isinstance(layer_name, tuple) else layer_name
        axes[idx].plot(prune_percs, pruned_acc_list, label=f"{beta}", marker="o")
        axes[idx].set_xlabel("Pruned %")
        axes[idx].set_ylabel("Test Loss")
        axes[idx].set_title(f"Layer: {layer_label}")
        axes[idx].grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for beta, lr in tqdm(betas):
    weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
    prune_model_per_layer(weights, axes, beta)

for ax in axes:
    ax.legend(title="β", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.invert_yaxis()
    #ax.set_yscale("log")

plt.tight_layout()
plt.savefig(f"plots/vib_mnist_beta_vs_pruned_loss_per_layer.png", dpi=300, bbox_inches="tight")
plt.show(block=block_plt)

# ------------------------------------------------------------------------------

def hist_sort_per_layer(weights, beta, all_weights_by_layer):
    model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
    model.load_state_dict(weights)

    for idx, layer_name in enumerate(layer_names):
        weight_values = []

        # Handle tuple layer names
        if isinstance(layer_name, tuple):
            for sub_layer in layer_name:
                module = dict(model.named_modules()).get(sub_layer)
                if module is not None and hasattr(module, "weight") and module.weight is not None:
                    weight_values.append(module.weight.data.flatten())
        else:
            module = dict(model.named_modules()).get(layer_name)
            if module is not None and hasattr(module, "weight") and module.weight is not None:
                weight_values.append(module.weight.data.flatten())

        if weight_values:
            layer_tensor = torch.cat(weight_values)
            all_weights_by_layer[idx].append(layer_tensor.abs().cpu().numpy())

all_weights_by_layer = [[] for _ in range(len(layer_names))]
beta_labels = []

for beta, lr in tqdm(betas):
    weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
    hist_sort_per_layer(weights, beta, all_weights_by_layer)
    if len(beta_labels) < len(betas) + 1:
        beta_labels.append(str(beta))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, layer_name in enumerate(layer_names):
    layer_label = ', '.join(layer_name) if isinstance(layer_name, tuple) else layer_name

    axes[idx].boxplot(
        all_weights_by_layer[idx],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="black"),
        medianprops=dict(color="red"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.6)
    )
    axes[idx].set_xticks(range(1, len(beta_labels) + 1))
    axes[idx].set_xticklabels(beta_labels, rotation=45)
    axes[idx].set_xlabel("Beta (β)")
    axes[idx].set_ylabel("Weight Values")
    axes[idx].set_title(f"Layer: {layer_label}")
    axes[idx].grid(True, axis="y", alpha=0.3)
    axes[idx].set_yscale("log")

plt.tight_layout()
plt.savefig(f"plots/vib_mnist_beta_weight_dist_per_layer.png", dpi=300, bbox_inches="tight")
plt.show(block=True)

# ------------------------------------------------------------------------------

"""
plt.figure(figsize=(10, 6))

for weights, label in zip(all_weights_by_beta, beta_labels):
    plt.hist(
        weights,
        bins=150,
        range=(0, 0.5),
        density=True, # normalize so we compare shape, not raw counts
        histtype="step",
        linewidth=2,
        label=f"Beta: {label}",
        log=True
    )

plt.xlabel("Weight Value")
plt.ylabel("Density (Log Scale)")
plt.title(f"Weight Distribution (Log Scale) - VIB MNIST ['{', '.join(layer_names)}']")
plt.legend(loc="upper right")
plt.grid(True, which="both", ls="-", alpha=0.2) # "both" grids for log scale

plt.tight_layout()
plt.savefig(f"plots/vib_mnist_beta_weight_hist_{'_'.join(layer_names)}.png", dpi=300, bbox_inches="tight")
plt.show(block=True)
"""
