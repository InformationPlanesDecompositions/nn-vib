import os, gzip, pickle, json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
            ax.text(
                x,
                y,
                f"({x:.4f}, {y:.2f})",
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

            if ys2:
                for x, y in zip(xs, ys2):
                    ax.text(
                        x,
                        y,
                        f"({x:.4f}, {y:.2f})",
                        fontsize=8,
                        verticalalignment="top",
                        horizontalalignment="left",
                    )

    ax.legend()
    return fig, ax


def get_device():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


@dataclass
class VIBNetParams:
    model_name: str
    beta: float
    z_dim: int
    hidden1: int
    hidden2: int
    lr: float
    batch_size: int
    epochs: int
    device: torch.device
    rnd_seed: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace, model_name: str):
        return cls(
            model_name=model_name,
            beta=args.beta,
            z_dim=args.z_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=get_device(),
            rnd_seed=args.rnd_seed,
        )

    def file_name(self) -> str:
        return f"vib_{self.model_name}_{self.hidden1}_{self.hidden2}_{self.z_dim}_{self.beta}_{self.lr}_{self.epochs}"

    def save_dir(self) -> str:
        s = f"save_stats_weights/{self.file_name()}"
        os.makedirs(s, exist_ok=True)
        return f"{s}/{self.file_name()}"

    def to_json(self, test_losses, test_accuracy, ce_kls):
        return {
            "test_losses": test_losses,
            "test_acc": test_accuracy,
            "ce_kls": ce_kls,
            "beta": self.beta,
            "z_dim": self.z_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "rnd_seed": self.rnd_seed,
        }

    def __str__(self):
        return (
            f"hyperparameters:\n"
            f"\tmodel         = {self.model_name}\n"
            f"\tbeta          = {self.beta}\n"
            f"\tz_dim         = {self.z_dim}\n"
            f"\thidden1       = {self.hidden1}\n"
            f"\thidden2       = {self.hidden2}\n"
            f"\tlr            = {self.lr}\n"
            f"\tepochs        = {self.epochs}\n"
            f"\tbatch_size    = {self.batch_size}\n"
            f"\tdevice        = {self.device}\n"
            f"\trnd_seed      = {self.rnd_seed}\n"
            f"\tsave_dir      = {self.save_dir()}"
        )


def load_weights(filepath, verbose=True):
    weights = torch.load(filepath, map_location="cpu")
    if verbose:
        print(f"loaded object type: {type(weights)}")
    if isinstance(weights, dict):
        if verbose:
            print("keys in the weights file:")
        for key in weights.keys():
            if verbose:
                print(f"- {key} with shape {weights[key].shape}")

    return weights


def plot_information_plane(ces: List[float], kls: List[float], save_dir: str):
    assert len(ces) == len(kls)

    i_x_t = np.array(kls)
    i_t_y = np.array(ces)
    epochs = np.arange(len(i_x_t))
    num_epochs = len(epochs)

    plt.figure(figsize=(12, 8))
    cmap_name = "viridis"
    cmap_instance = plt.get_cmap(cmap_name)

    norm = plt.Normalize(vmin=epochs.min(), vmax=epochs.max())

    for i in range(num_epochs - 1):
        color = cmap_instance(norm(epochs[i]))
        plt.plot(
            i_x_t[i : i + 2],
            i_t_y[i : i + 2],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            zorder=1,
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
        zorder=2,
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


# TODO: plot ce, kl, loss for both test and train so 2 sub plots
def plot_losses(
    test_losses: List[float],
    train_losses: List[float],
    file_name: str,
    save_dir: str,
) -> None:
    epochs = len(test_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, epochs + 1),
        train_losses,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#1f77b4",
        label="training Loss",
    )
    plt.plot(
        range(1, epochs + 1),
        test_losses,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#ff7f0e",
        label="test Loss",
    )
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


def idx_extractor(filepath: str) -> np.ndarray:
    open_func = gzip.open if filepath.endswith(".gz") else open
    with open_func(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_dimensions = magic % 256
        dimensions = []
        for _ in range(num_dimensions):
            dimensions.append(int.from_bytes(f.read(4), "big"))
        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data.reshape(dimensions)


class FashionMnistIdxDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True):
        prefix = "train" if train else "t10k"

        images_filepath = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte")
        labels_filepath = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte")

        if not os.path.exists(images_filepath) or not os.path.exists(labels_filepath):
            raise FileNotFoundError(
                f"Could not find required files in '{data_dir}'. "
                f"Expected: {os.path.basename(images_filepath)} and {os.path.basename(labels_filepath)}"
            )

        images_np = idx_extractor(images_filepath)
        labels_np = idx_extractor(labels_filepath)
        self.images = torch.from_numpy(images_np.copy()).float().div(255.0)
        self.labels = torch.from_numpy(labels_np.copy()).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx].view(1, 28, 28)
        label = self.labels[idx]
        return image, label


class CIFAR10Dataset(Dataset):
    """https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"""

    def __init__(self, data_dir: str, train: bool = True):
        self.data_dir = data_dir
        batch_files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        image_batches, label_batches = [], []

        for batch_file in batch_files:
            batch_path = os.path.join(self.data_dir, batch_file)
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"Could not find required file '{batch_file}' in '{self.data_dir}'")
            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            data = batch[b"data"] if b"data" in batch else batch["data"]
            labels = batch[b"labels"] if b"labels" in batch else batch["labels"]
            image_batches.append(data)
            label_batches.append(labels)

        images_np = np.vstack(image_batches).astype(np.float32)
        labels_np = np.array(sum(label_batches, []), dtype=np.int64)
        images_np = images_np.reshape(-1, 3, 32, 32) / 255.0

        self.images = torch.from_numpy(images_np.copy())
        self.labels = torch.from_numpy(labels_np.copy())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def weights_location(h1, h2, z_dim, beta, lr):
    top_dir = "save_stats_weights"
    var = lambda v, w, x, y, z: f"vib_mnist_{v}_{w}_{x}_{y}_{z}"
    return f"{top_dir}/{var(h1, h2, z_dim, beta, lr)}/{var(h1, h2, z_dim, beta, lr)}.pth"


def vib_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, y)
    variance = sigma.pow(2)
    log_variance = 2 * torch.log(sigma)
    kl_terms = 0.5 * (variance + mu.pow(2) - 1.0 - log_variance)
    kl = torch.sum(kl_terms, dim=1).mean()
    total_loss = ce + beta * kl
    return ce, kl, total_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits, mu, sigma = model(X)
        _, _, loss = vib_loss(logits, Y, mu, sigma, beta)
        loss.backward()
        optimizer.step()
        bs = X.size(0)
        running_loss += loss.item() * bs
        _, preds = torch.max(logits, 1)
        correct += (preds == Y).sum().item()
        total += bs
        tq.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}",
            }
        )
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    mc_samples: int = 10,
) -> Tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        batches = list(dataloader)
        Xs, Ys = zip(*batches)
        X = torch.cat(Xs, dim=0).to(device)
        Y = torch.cat(Ys, dim=0).to(device)
        logits, mu, sigma = model(X)
        probs_sum = F.softmax(logits, dim=1)
        for _ in range(mc_samples - 1):
            logits, _, _ = model(X)
            probs_sum += F.softmax(logits, dim=1)
        probs = probs_sum / mc_samples
        ce = F.nll_loss(torch.log(probs.clamp_min(1e-8)), Y)
        variance = sigma.pow(2)
        log_variance = 2 * torch.log(sigma)
        kl_terms = 0.5 * (variance + mu.pow(2) - 1.0 - log_variance)
        kl = torch.sum(kl_terms, dim=1).mean()
        loss = ce + beta * kl
        _, preds = torch.max(probs, 1)
        acc = 100.0 * (preds == Y).float().mean().item()
    return loss.item(), ce.item(), kl.item(), acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    beta: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model.to(device)
    train_losses, test_losses, test_ces, test_kls = [], [], [], []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=beta,
        )
        test_loss, ce, kl, test_acc = evaluate_epoch(
            model,
            test_loader,
            device,
            beta=beta,
        )
        print(
            f"""epoch [{epoch + 1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
\t\t\ttest loss: {test_loss:.3f} | test acc: {test_acc:.2f}%"""
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_ces.append(ce)
        test_kls.append(kl)
    return train_losses, test_losses, test_ces, test_kls


def run_training_job(
    model: nn.Module,
    params: VIBNetParams,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> None:
    if not params.rnd_seed:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    print(params)
    train_loader = DataLoader(train_dataset, params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, params.batch_size, shuffle=False)
    print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))
    train_losses, test_losses, test_ces, test_kls = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        params.device,
        params.epochs,
        beta=params.beta,
    )
    test_loss, _, _, test_acc = evaluate_epoch(model, test_loader, params.device, params.beta)
    print(f"lr: {params.lr}, test loss: {test_loss}, test acc: {test_acc}")
    torch.save(model.state_dict(), f"{params.save_dir()}.pth")
    with open(f"{params.save_dir()}_stats.json", "w") as json_file:
        json.dump(
            params.to_json(test_losses, test_acc, list(zip(test_ces, test_kls))),
            json_file,
            indent=2,
        )
    plot_losses(test_losses, train_losses, params.file_name(), params.save_dir())
    plot_information_plane(test_ces, test_kls, params.save_dir())
