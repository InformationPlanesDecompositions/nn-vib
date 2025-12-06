#!/usr/bin/env python3

import json
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from msc import get_device

class MnistCsvDataset(Dataset):
    def __init__(self, filepath: str):
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.images = torch.tensor(data[:, 1:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx].view(1, 28, 28), self.labels[idx]

@dataclass
class VIBNetParams:
    beta: float
    z_dim: int
    hidden1: int
    hidden2: int

    lr: float
    lr_decay: bool
    batch_size: int
    epochs: int

    device: torch.device

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
                beta=args.beta,
                z_dim = args.z_dim,
                hidden1 = args.hidden1,
                hidden2 = args.hidden2,
                lr = args.lr,
                lr_decay = args.lr_decay,
                batch_size = args.batch_size,
                epochs = args.epochs,
                device = get_device(),
        )

    def file_name(self) -> str:
        return f"vib_mnist_{self.hidden1}_{self.hidden2}_{self.z_dim}_{self.beta}_{self.lr}"

    def save_dir(self) -> str:
        s = f"save_stats_weights/{self.file_name()}"
        os.makedirs(s, exist_ok=True)
        return f"{s}/{self.file_name()}"

    def to_json(self, test_losses: List[float], test_accuracy: float) -> Dict:
        return {
                "test_losses": test_losses,
                "test_acc": test_accuracy,

                "beta": self.beta,
                "z_dim": self.z_dim,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,

                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
        }

    def __str__(self):
        return (
                f"hyperparameters:\n"
                f"\tbeta          = {self.beta}\n"
                f"\tz_dim         = {self.z_dim}\n"
                f"\thidden1       = {self.hidden1}\n"
                f"\thidden2       = {self.hidden2}\n"
                f"\tlr            = {self.lr}\n"
                f"\tepochs        = {self.epochs}\n"
                f"\tbatch_size    = {self.batch_size}\n"
                f"\tdevice        = {self.device}\n"
                f"\tsave_dir      = {self.save_dir()}"
        )

class VIBNet(nn.Module):
    def __init__(
            self,
            z_dim: int,
            input_shape: int,
            hidden1: int,
            hidden2: int,
            output_shape: int,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, hidden1)

        self.fc_mu = nn.Linear(hidden1, z_dim)
        self.fc_logvar = nn.Linear(hidden1, z_dim)

        self.fc2 = nn.Linear(z_dim, hidden2)
        self.fc_decode = nn.Linear(hidden2, output_shape)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # deep vib eq. 19
        sigma = F.softmax(logvar - 5.0, dim=1)

        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor):
        h = F.relu(self.fc2(x))
        logits = self.fc_decode(h)
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        mu, sigma = self.encode(x_flat)
        z = self.reparameterize(mu, sigma)
        logits = self.decode(z)
        return logits, mu, sigma

def vib_loss(
        logits: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        beta: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    kl: upper bound on I(Z;X) (compression)
    ce: proxy for H(Y|Z) (relevance term)
    # (beta bigger = more compression)
    """
    ce = F.cross_entropy(logits, y)

    sigma = torch.clamp(sigma, 1e-10)
    log_sigma_sq = 2 * torch.log(sigma)
    kl_terms = 0.5 * (sigma.pow(2) + mu.pow(2) - 1.0 - log_sigma_sq)
    kl = torch.sum(kl_terms, dim=1).mean()

    total_loss = ce + beta*kl
    return ce, kl, total_loss

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        beta: float
) -> Tuple[float, float, float, float]:
    model.train()

    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0

    correct = 0
    total = 0

    for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        logits, mu, sigma = model(X)
        ce, kl, loss = vib_loss(logits, Y, mu, sigma, beta)
        loss.backward()
        optimizer.step()

        # accumulate (note multiply nll by batch size to match previous running_loss scheme)
        bs = X.size(0)
        running_loss += loss.item() * bs
        running_ce += ce.item()
        running_kl += kl.item()

        _, preds = torch.max(logits, 1)
        correct += (preds == Y).sum().item()
        total += bs

        tq.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}"
        })

    avg_loss = running_loss / total
    avg_ce = running_ce / total
    avg_kl = running_kl / total
    accuracy = 100.0 * correct / total
    return avg_loss, avg_ce, avg_kl, accuracy

def evaluate_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        beta: float
) -> Tuple[float, float, float, float]:
    model.eval()

    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, mu, sigma = model(images)
            ce, kl, loss = vib_loss(logits, labels, mu, sigma, beta)

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_ce += ce.item()
            running_kl += kl.item()

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += bs

    avg_loss = running_loss / total
    avg_ce = running_ce / total
    avg_kl = running_kl / total
    accuracy = 100.0 * correct / total

    return avg_loss, avg_ce, avg_kl, accuracy

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        device: torch.device,
        epochs: int,
        beta: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model.to(device)
    train_losses, test_losses, test_ces, test_kls = [], [], [], []
    for epoch in range(epochs):
        train_loss, _, _, train_acc = train_epoch(model, train_loader, optimizer, device, beta=beta)
        test_loss, test_ce, test_kl, test_acc = evaluate_epoch(model, test_loader, device, beta=beta)
        if scheduler:
            scheduler.step()
            print(f"lr: {optimizer.param_groups[0]['lr']:.10f}")

        print(f"""epoch [{epoch+1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
                    test loss: {test_loss:.3f} | test acc: {test_acc:.2f}%""")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_ces.append(test_ce)
        test_kls.append(test_kl)

    return train_losses, test_losses, test_ces, test_kls

def main() -> None:
    parser = argparse.ArgumentParser(description="training script with configurable hyperparameters.")
    parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
    parser.add_argument("--z_dim", type=int, default=75, help="latent dimension size")
    parser.add_argument("--hidden1", type=int, default=300, help="size of first hidden layer")
    parser.add_argument("--hidden2", type=int, default=100, help="size of second hidden layer")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs") # 200 in deep vib paper
    parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate") # 1e-4 in deep vib paper
    parser.add_argument("--lr_decay", type=bool, default=False, help="Enable learning rate decay")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size") # 100 in deep vib paper
    args = parser.parse_args()

    if not args.rnd_seed:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    params = VIBNetParams.from_args(args)
    print(params)

    dataset = MnistCsvDataset("data/mnist_data.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f"train set size: {train_size}, test set size: {test_size}")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, params.batch_size, shuffle=True)

    model = VIBNet(params.z_dim, 784, params.hidden1, params.hidden2, 10)
    print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))
    scheduler = None
    if params.lr_decay:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.94 ** (epoch // 2))

    train_losses, test_losses, test_ces, test_kls = train_model(
            model,
            train_loader, test_loader,
            optimizer, scheduler,
            params.device,
            params.epochs,
            beta=params.beta,
    )

    test_loss, _, _, test_acc = evaluate_epoch(model, test_loader, params.device, params.beta)
    print(f"test loss: {test_loss}, test acc: {test_acc}")

    torch.save(model.state_dict(), f"{params.save_dir()}.pth")
    with open(f"{params.save_dir()}_stats.json", "w") as json_file:
        json.dump(params.to_json(test_losses, test_acc), json_file, indent=2)

    plot_losses(test_losses, train_losses, params)
    plot_information_plane(test_ces, test_kls, params)

def plot_losses(test_losses: List[float], train_losses: List[float], params: VIBNetParams) -> None:
    epochs = len(test_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", linewidth=2, markersize=6, color="#1f77b4", label="training Loss")
    plt.plot(range(1, epochs + 1), test_losses, marker="o", linewidth=2, markersize=6, color="#ff7f0e", label="test Loss")
    plt.title(f"({params.file_name()}) loss", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f"{params.save_dir()}_test_loss.png", dpi=300, bbox_inches="tight")

def plot_information_plane(ces: List[float], kls: List[float], params: VIBNetParams):
    epochs = len(kls)
    epoch_numbers = np.arange(1, epochs + 1) # Array of epoch numbers for coloring

    plt.figure(figsize=(12, 9))

    # --- 1. Plot the Training Trajectory (Color Gradient) ---
    # We use a scatter plot, connecting the dots with a line for the trajectory visualization.
    # The "c" parameter takes the epoch number, and "cmap" defines the color map.

    # 1a. Plot the colored dots
    scatter = plt.scatter(
        kls,
        ces,
        c=epoch_numbers,          # Color based on epoch number
        cmap="jet",               # Choose a suitable colormap (e.g., "viridis", "jet", "plasma")
        s=80,                     # Size of the markers
        zorder=2                  # Ensure dots are above the line
    )

    # 1b. Plot the trajectory line
    plt.plot(
        kls,
        ces,
        linestyle="-",
        linewidth=1,
        color="gray",             # Use a neutral color for the connecting line
        alpha=0.5,
        zorder=1                  # Ensure line is below the dots
    )

    # --- 2. Highlight Start and End Points ---
    plt.scatter(kls[0], ces[0], color="green", s=150, label="Start (Epoch 1)", marker="o", edgecolors="black", zorder=3)
    plt.scatter(kls[-1], ces[-1], color="red", s=150, label="End (Epoch N)", marker="o", edgecolors="black", zorder=3)

    # --- 3. Add Colorbar Legend ---
    cbar = plt.colorbar(scatter)
    cbar.set_label("Epoch Number", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # --- 4. Plot Details and Formatting ---
    plt.title(f"({params.file_name()}) Information Plane (β={params.beta})", fontsize=18, fontweight="bold", pad=20)
    # Note: KL is an upper bound on I(X;Z) (Compression) and CE is a proxy for I(Z;Y) (Distortion/Relevance).
    plt.xlabel("Compression: KL Divergence $\\mathbb{I}(X;Z)$ Upper Bound", fontsize=15)
    plt.ylabel("Distortion: Cross-Entropy $\\mathbb{I}(Z;Y)$ Proxy", fontsize=15)

    # Information Plane is often plotted with I(Z;Y) increasing upwards (lower CE), so we invert the Y-axis.
    plt.gca().invert_yaxis()

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{params.save_dir()}_info_plane.png", dpi=300)
    plt.close()

if __name__ == "__main__": main()
