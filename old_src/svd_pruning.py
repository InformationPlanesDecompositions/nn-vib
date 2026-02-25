#!/usr/bin/env python3

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from vib_mnist_train import VIBNet, evaluate_epoch, FashionMnistIdxDataset
from msc import get_device, load_weights, weights_location
from svd import svd

torch.manual_seed(42)
device = get_device()
print(f"using device: {device}")

block_plt = False
batch_size = 100

train_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=True)
test_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=False)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

betas = [
    (0.4, 0.0001),
    (0.3, 0.0001),
    (0.2, 0.0001),
    (0.1, 0.0005),
    (0.05, 0.0005),
    (0.01, 0.001),
    (0.005, 0.0005),
    (0.001, 0.0005),
    (0.0, 0.0001),
]
z_dim, h1, h2, o_shape = 125, 500, 300, 10

layer_names = ["fc1", "fc_mu", "fc_logvar", "fc2", "fc_decode"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# finds the rank k that preserves energy_threshold of the matrix energy.
def get_pruning_threshold(s_vals, energy_threshold: float):
    assert energy_threshold <= 1.0
    if energy_threshold <= 0:
        return 1
    # For SVD, per-component "energy" is proportional to sigma^2.
    # Example: energy_threshold=0.90 means keep the smallest rank k whose
    # cumulative sigma^2 reaches 90% of the layer's total sigma^2.
    energy = s_vals**2
    total_energy = torch.sum(energy)
    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
    mask = cumulative_energy >= (energy_threshold - 1e-7)
    indices = torch.where(mask)[0]
    if len(indices) == 0:
        return len(s_vals)
    k = indices[0].item() + 1
    return k


class LowRankLinear(nn.Module):
    """Replaces nn.Linear with low-rank factorization"""

    def __init__(self, U, S, Vh, k, bias=None):  # Added k parameter!
        super().__init__()
        # Absorb sqrt(S) into both factors for numerical stability
        sqrt_S = torch.sqrt(S[:k])
        self.U_scaled = nn.Parameter(U[:, :k] * sqrt_S)  # [M, k]
        self.V_scaled = nn.Parameter(Vh[:k, :] * sqrt_S.unsqueeze(1))  # [k, N]
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        # x: [batch, N] -> x @ V.t(): [batch, N] @ [N, k] = [batch, k]
        # -> @ U.t(): [batch, k] @ [k, M] = [batch, M]
        out = x @ self.V_scaled.t() @ self.U_scaled.t()
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------- SVD PRUNE WHOLE NETWORK ------------------------

beta, lr = 0.001, 0.0005
# beta, lr = 0.0, 0.0001
weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
model.load_state_dict(weights)
train_loss, ce, kl, train_acc = evaluate_epoch(model, train_loader, device, beta=beta)
test_loss, ce, kl, test_acc = evaluate_epoch(model, test_loader, device, beta=beta)
og_param_count = count_params(model)
print(f"size of model before prune: {og_param_count}")
print(f"original: train_loss = {train_loss:.3f}, train_acc = {train_acc:.2f}%")
print(f"original: test_loss = {test_loss:.3f}, test_acc = {test_acc:.2f}%")
print()

# Thresholds are "variance retained" per layer (not fraction of weights kept).
# So 0.90 keeps enough singular directions to retain 90% energy; 0.60 is more aggressive.
layer_names_w_thresh = [("fc1", 1.0), ("fc_mu", 0.95), ("fc_logvar", 0.9), ("fc2", 0.65), ("fc_decode", 1.0)]
svd_results = svd(model, layer_names)
for idx, (layer_name, thresh) in enumerate(layer_names_w_thresh):
    U = svd_results[layer_name]["U"]
    S = svd_results[layer_name]["S"]
    Vh = svd_results[layer_name]["Vh"]  # V-hermitian/transpose
    k = get_pruning_threshold(S, energy_threshold=thresh)

    module = dict(model.named_modules())[layer_name]
    M, N = module.weight.shape

    # calculate if pruning actually saves parameters
    original_params = M * N
    low_rank_params = M * k + k * N

    # only prune if it reduces parameters
    if low_rank_params >= original_params:
        continue

    bias = module.bias.data if module.bias is not None else None

    # create low-rank replacement with explicit k
    low_rank_module = LowRankLinear(U, S, Vh, k, bias).to(device)

    # replace in model
    parent = model
    attr_name = layer_name
    setattr(parent, attr_name, low_rank_module)

    savings = original_params - low_rank_params
    print(
        f"layer {layer_name}: pruned rank from {len(S)} to {k} (saved {savings:,} params, {100 * savings / original_params:.1f}%)"
    )

n_param_count = count_params(model)
param_perc_diff = (og_param_count - n_param_count) / og_param_count * 100
train_loss, ce, kl, train_acc = evaluate_epoch(model, train_loader, device, beta=beta)
test_loss, ce, kl, test_acc = evaluate_epoch(model, test_loader, device, beta=beta)
print()
print(f"size of model after prune: {n_param_count}, pruned: {param_perc_diff:.2f}% of total weights")
print(f"original: train_loss = {train_loss:.3f}, train_acc = {train_acc:.2f}%")
print(f"pruned: test_loss = {test_loss:.3f}, test_acc = {test_acc:.2f}%")

# exit(1)

# ------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, layer in enumerate(layer_names):
    ax = axes[idx]
    for beta, lr in betas:
        weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
        model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
        model.load_state_dict(weights)
        out = svd(model, layer_names)

        # the singular values that represent the strength or importance of
        #   different directions in the data
        S = out[layer]["S"].cpu().numpy()

        ax.plot(S, label=f"β: {beta}", alpha=0.8, linewidth=2.5)

    ax.set_title(f"Layer: {layer}")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Magnitude")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

# gide the extra subplot (we have 5 layers in a 2x3 grid)
axes[5].axis("off")

plt.suptitle("Singular Value Spectrum Comparison Across Betas", fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(f"plots/vib_mnist_beta_svd.png", dpi=300, bbox_inches="tight")
# what we are seeing here:
#   If only a few singular values are large and the rest are near zero,
#   the layer is low-rank. This means the layer is over-parameterized—it
#   has more neurons than it actually needs to represent the
#   transformation it has learned.
# If a layer is low rank, you can compress is significantly without
#   losing much accuracy.
plt.show()

# ---------------------- SVD PRUNING ----------------------

# We prune by retained SVD energy (sum of kept sigma^2 / total sigma^2).
# 1.0 ~= no pruning; 0.6 means keep enough components to explain 60% variance.
energy_levels = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.80]


def svd_prune_model_per_layer(weights_path, axes, beta):
    model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
    model.load_state_dict(weights_path)

    svd_results = svd(model, layer_names)

    for idx, layer_name in enumerate(layer_names):
        pruned_loss_list = []

        current_layers = [layer_name] if isinstance(layer_name, str) else layer_name

        for energy_threshold in energy_levels:
            pruned_model = copy.deepcopy(model)

            for sub_layer in current_layers:
                U = svd_results[sub_layer]["U"]
                S = svd_results[sub_layer]["S"]
                Vh = svd_results[sub_layer]["Vh"]  # V-hermitian/transpose

                k = get_pruning_threshold(S, energy_threshold=energy_threshold)

                print(f"layer {sub_layer}: pruned rank from {len(S)} to {k}")

                # 3. Reconstruct Low-Rank Matrix: U_k * S_k * Vh_k
                # This keeps the matrix shape [M, N] so VIBNet still works
                W_reconstructed = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

                module = dict(pruned_model.named_modules())[sub_layer]
                with torch.no_grad():
                    module.weight.copy_(W_reconstructed)

            test_loss, ce, kl, test_acc = evaluate_epoch(pruned_model, test_loader, device, beta=beta)
            pruned_loss_list.append(test_loss)

        layer_label = ", ".join(layer_name) if isinstance(layer_name, tuple) else layer_name
        # X-axis is "1 - energy" to represent "Amount Pruned"
        prune_percs = [1.0 - e for e in energy_levels]
        axes[idx].plot(prune_percs, pruned_loss_list, label=f"β={beta}", marker="o")
        axes[idx].set_xlabel("Energy Pruned (1 - Variance Retained)")
        axes[idx].set_ylabel("Test Loss")
        axes[idx].set_title(f"SVD Pruning: {layer_label}")
        axes[idx].grid(True, alpha=0.3)


num_layers = len(layer_names)
cols = 2
rows = (num_layers + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = axes.flatten()
for i in range(num_layers, len(axes)):
    axes[i].axis("off")

for beta, lr in tqdm(betas):
    weights = load_weights(weights_location(h1, h2, z_dim, beta, lr), verbose=False)
    svd_prune_model_per_layer(weights, axes, beta)

for ax in axes:
    ax.legend(title="β", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"plots/vib_mnist_beta_svd_pruning.png", dpi=300, bbox_inches="tight")
plt.show()
