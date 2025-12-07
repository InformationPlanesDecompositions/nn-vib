from typing import List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def plot_x_y(
        xs: List,
        ys1: List,
        ys2: Optional[List],
        xlabel: str,
        ylabel: str,
        line_1_label: str,
        line_2_label: Optional[str],
        xlog: bool,
        point_labels: bool,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(xs, ys1, marker="o", linestyle="-", color="b", label=line_1_label)
    if ys2 and line_2_label:
        ax.plot(xs, ys2, marker="x", linestyle="--", color="r", label=line_2_label)

    if xlog:
        ax.set_xscale("log")

    ax.set_title(f"{ylabel} vs {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if point_labels:
        for x, y in zip(xs, ys1):
            ax.text(x, y, f"({x:.4f}, {y:.2f})", fontsize=8,
                            verticalalignment="bottom", horizontalalignment="right")

        if ys2:
            for x, y in zip(xs, ys2):
                ax.text(x, y, f"({x:.4f}, {y:.2f})", fontsize=8,
                                verticalalignment="top", horizontalalignment="left")

    ax.legend()
    return fig, ax

def get_device():
    device = ""
    if torch.cuda.is_available(): device = "cuda"
    elif torch.mps.is_available(): device = "mps"
    else: device = "cpu"
    return torch.device(device)

def load_weights(filepath, verbose=True):
    weights = torch.load(filepath, map_location="cpu")
    if verbose: print(f"loaded object type: {type(weights)}")
    if isinstance(weights, dict):
        if verbose: print("keys in the weights file:")
        for key in weights.keys():
            if verbose: print(f"- {key} with shape {weights[key].shape}")

    return weights

def plot_information_plane(ces: List[float], kls: List[float], save_dir: str):
    assert len(ces) == len(kls)

    i_x_t = np.array(kls)
    i_t_y = np.array([-ce for ce in ces])
    epochs = np.arange(len(i_x_t))
    num_epochs = len(epochs)

    plt.figure(figsize=(12, 8))
    cmap_name = "viridis"
    cmap_instance = plt.get_cmap(cmap_name)

    norm = plt.Normalize(vmin=epochs.min(), vmax=epochs.max())

    for i in range(num_epochs - 1):
        color = cmap_instance(norm(epochs[i]))
        plt.plot(
            i_x_t[i:i+2],
            i_t_y[i:i+2],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            zorder=1
        )

    scatter = plt.scatter(
        i_x_t,
        i_t_y,
        c=epochs,
        cmap=cmap_name,
        norm=norm,
        marker="o",
        s=50,
        alpha=1.0,
        zorder=2
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("epoch", rotation=270, labelpad=15, fontsize=12)
    plt.xscale("log")
    plt.title(f"Information Plane", fontsize=16)
    plt.xlabel("I(X;Z)", fontsize=14)
    plt.ylabel("I(Z;Y)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"{save_dir}_info_plane.png", dpi=300)
    plt.close()

def plot_losses(test_losses: List[float], train_losses: List[float], file_name: str, save_dir: str) -> None:
    epochs = len(test_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", linewidth=2, markersize=6, color="#1f77b4", label="training Loss")
    plt.plot(range(1, epochs + 1), test_losses, marker="o", linewidth=2, markersize=6, color="#ff7f0e", label="test Loss")
    plt.title(f"({file_name}) loss", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f"{save_dir}_test_loss.png", dpi=300, bbox_inches="tight")

class MnistCsvDataset(Dataset):
    def __init__(self, filepath: str):
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.images = torch.tensor(data[:, 1:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx].view(1, 28, 28), self.labels[idx]
