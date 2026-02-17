#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from msc import get_device, load_weights, weights_location, MnistCsvDataset
from vib_mnist_train import VIBNet

DEVICE = get_device()

def extract_features(model, data_loader, target_layer_name):
    model.eval()
    activations = {}

    def get_activations(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    target_layer = dict(model.named_modules())[target_layer_name]
    h = target_layer.register_forward_hook(get_activations("target"))

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            X = images.to(DEVICE)
            _ = model(X)
            current_activations = activations["target"].numpy()
            all_features.append(current_activations)
            all_labels.append(batch_labels.cpu().numpy())

    h.remove()

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    return features_np, labels_np

def perform_pca_and_plot(ax, features, labels, title):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    pca_df["Target_Label"] = labels

    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices_to_keep = pca_df["Target_Label"] == label

        ax.scatter(
            pca_df.loc[indices_to_keep, "PC1"],
            pca_df.loc[indices_to_keep, "PC2"],
            label=f"Class {int(label)}",
            alpha=0.6,
            s=10
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)


betas = [(0.02, 0.0001), (0.01, 5e-05), (0.005, 5e-05), (0.001, 5e-05), (0.0005, 5e-05), (0.0001, 1e-05)]
z_dim, h1, h2, o_shape = 75, 300, 100, 10

dataset = MnistCsvDataset("data/mnist_data.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
print(f"train set size: {train_size}, test set size: {test_size}")
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, 100, shuffle=True)

layer_to_inspect = "fc_mu"

num_plots = len(betas)+1
cols = 3
rows = (num_plots + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten()

for i, (beta, lr) in tqdm(enumerate(betas)):
    weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
    model = VIBNet(z_dim, 784, h1, h2, o_shape).to(DEVICE)
    model.load_state_dict(weights)

    features, labels = extract_features(model, test_loader, layer_to_inspect)
    perform_pca_and_plot(axes[i], features, labels, f"β: {beta}, lr: {lr}")

weights = load_weights("save_stats_weights/vib_mnist_300_100_75_0.01_0.0001_500/vib_mnist_300_100_75_0.01_0.0001.pth", verbose=False)
model = VIBNet(z_dim, 784, h1, h2, o_shape).to(DEVICE)
model.load_state_dict(weights)
features, labels = extract_features(model, test_loader, layer_to_inspect)
perform_pca_and_plot(axes[6], features, labels, f"β: 0.01, lr: 0.0001 500")

for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", title="MNIST Class", fontsize=9)

plt.suptitle("pca visualization of latent space vib mnist models", fontsize=14, y=1.02)
plt.savefig(f"plots/vib_mnist_per_beta_pca.png", dpi=300, bbox_inches="tight")
plt.show(block=True)
