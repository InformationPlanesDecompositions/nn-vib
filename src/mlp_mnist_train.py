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
from torch.utils.data import random_split, DataLoader
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np
from msc import get_device, plot_information_plane, plot_losses, MnistCsvDataset

@dataclass
class MLPNetParams:
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
                hidden1 = args.hidden1,
                hidden2 = args.hidden2,
                lr = args.lr,
                lr_decay = args.lr_decay,
                batch_size = args.batch_size,
                epochs = args.epochs,
                device = get_device(),
        )

    def file_name(self) -> str:
        return f"mlp_mnist_{self.hidden1}_{self.hidden2}_{self.lr}"

    def save_dir(self) -> str:
        s = f"save_stats_weights/{self.file_name()}"
        os.makedirs(s, exist_ok=True)
        return f"{s}/{self.file_name()}"

    def to_json(self, test_losses: List[float], test_accuracy: float, ce_kls: List[Tuple[float, float]]) -> Dict:
        return {
                "test_losses": test_losses,
                "test_acc": test_accuracy,
                "ce_kls": ce_kls,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
        }

    def __str__(self):
        return (
                f"hyperparameters:\n"
                f"\thidden1       = {self.hidden1}\n"
                f"\thidden2       = {self.hidden2}\n"
                f"\tlr            = {self.lr}\n"
                f"\tepochs        = {self.epochs}\n"
                f"\tbatch_size    = {self.batch_size}\n"
                f"\tdevice        = {self.device}\n"
                f"\tsave_dir      = {self.save_dir()}"
        )

class MLPNet(nn.Module):
    def __init__(
            self,
            input_shape: int,
            hidden1: int,
            hidden2: int,
            output_shape: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_decode = nn.Linear(hidden2, output_shape)

    def encode(self, x: torch.Tensor):
        T = F.relu(self.fc1(x))
        return T

    def decode(self, x: torch.Tensor):
        h = F.relu(self.fc2(x))
        logits = self.fc_decode(h)
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        T = self.encode(x_flat)
        logits = self.decode(T)
        return T, logits

def mlp_loss(
    T: torch.Tensor,
    X: torch.Tensor, # original flattened input X
    logits: torch.Tensor,
    Y: torch.Tensor,
    num_bins: int = 30, # bin count for discretization
    sample_x_features: int = 20 # number of X features to sample for I(X;T)
) -> Tuple[torch.Tensor, float, float]:
    # 1. ACTUAL LOSS (Differentiable, PyTorch)
    ce_loss = F.cross_entropy(logits, Y)

    # Convert to NumPy for non-differentiable MI estimation
    T_np = T.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()

    # Simple check for reliability
    if X_np.shape[0] < num_bins:
        return ce_loss, 0.0, 0.0

    # --- 2. I(T;Y) Estimation (Relevance) - Component-wise Summation ---
    total_i_t_y_val = 0.0
    num_classes = len(np.unique(Y_np))

    for i in range(T_np.shape[1]):
        T_i = T_np[:, i]

        # Calculate 2D Joint Histogram H(T_i, Y)
        # Bins are defined by number of bins for T_i and number of classes for Y
        H_T_i_Y, _, _ = np.histogram2d(T_i, Y_np, bins=[num_bins, num_classes])

        P_T_i_Y = H_T_i_Y / np.sum(H_T_i_Y)
        P_T_i = np.sum(P_T_i_Y, axis=1) # Marginal P(T_i)
        P_Y = np.sum(P_T_i_Y, axis=0)  # Marginal P(Y)

        # Filter zeros for log calculation
        P_T_i_Y_nz = P_T_i_Y[P_T_i_Y > 0]; P_T_i_nz = P_T_i[P_T_i > 0]; P_Y_nz = P_Y[P_Y > 0]

        # Calculate entropies
        H_T_i_Y_val = entropy(P_T_i_Y_nz.flatten(), base=2)
        H_T_i_val = entropy(P_T_i_nz, base=2)
        H_Y_val = entropy(P_Y_nz, base=2)

        # I(T_i; Y) = H(T_i) + H(Y) - H(T_i, Y)
        I_T_i_Y_val = H_T_i_val + H_Y_val - H_T_i_Y_val
        total_i_t_y_val += I_T_i_Y_val

    # --- 3. I(X;T) Estimation (Compression) - X Sampling + T Mean Proxy ---
    total_i_x_t_val = 0.0
    X_sampled = X_np[:, :sample_x_features]
    T_mean = np.mean(T_np, axis=1) # 1D proxy for T

    # Pre-calculate H(T_mean)
    P_T_mean_all, _ = np.histogram(T_mean, bins=num_bins)
    P_T_mean_norm = P_T_mean_all / np.sum(P_T_mean_all)
    H_T_mean_val = entropy(P_T_mean_norm[P_T_mean_norm > 0], base=2)

    for i in range(X_sampled.shape[1]):
        X_i = X_sampled[:, i]

        # Calculate 2D Joint Histogram H(X_i, T_mean)
        # Using num_bins directly applies the same bin count to both axes
        H_X_i_T, _, _ = np.histogram2d(X_i, T_mean, bins=num_bins)

        # Calculate components
        P_X_i_T = H_X_i_T / np.sum(H_X_i_T)
        P_X_i = np.sum(P_X_i_T, axis=1) # Marginal P(X_i)

        P_X_i_T_nz = P_X_i_T[P_X_i_T > 0].flatten()
        P_X_i_nz = P_X_i[P_X_i > 0]

        H_X_i_T_val = entropy(P_X_i_T_nz, base=2)
        H_X_i_val = entropy(P_X_i_nz, base=2)

        # I(X_i; T_mean) = H(X_i) + H(T_mean) - H(X_i, T_mean)
        I_X_i_T_val = H_X_i_val + H_T_mean_val - H_X_i_T_val
        total_i_x_t_val += I_X_i_T_val

    return ce_loss, total_i_x_t_val, total_i_t_y_val

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float, float, float]:
    model.train()

    running_loss = 0.0
    running_i_x_t = 0.0
    running_i_t_y = 0.0

    correct = 0
    total = 0

    for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        T, logits = model(X)
        loss, I_X_T, I_T_Y = mlp_loss(T, X, logits, Y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs
        running_i_x_t += I_X_T
        running_i_t_y += I_T_Y

        _, preds = torch.max(logits, 1)
        correct += (preds == Y).sum().item()
        total += bs

        tq.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}"
        })

    avg_loss = running_loss / total
    avg_i_x_t = running_i_x_t / total
    avg_i_t_y = running_i_t_y / total
    accuracy = 100.0 * correct / total
    return avg_loss, avg_i_x_t, avg_i_t_y, accuracy

def evaluate_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()

    running_loss = 0.0
    running_i_x_t = 0.0
    running_i_t_y = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            X, Y = images.to(device), labels.to(device)
            T, logits = model(X)
            loss, I_X_T, I_T_Y = mlp_loss(T, X, logits, Y)

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_i_x_t += I_X_T
            running_i_t_y += I_T_Y

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += bs

    avg_loss = running_loss / total
    avg_i_x_t = running_i_x_t / total
    avg_i_t_y = running_i_t_y / total
    accuracy = 100.0 * correct / total

    return avg_loss, avg_i_x_t, avg_i_t_y, accuracy

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        device: torch.device,
        epochs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    model.to(device)
    train_losses, test_losses, test_ces, test_kls = [], [], [], []
    for epoch in range(epochs):
        train_loss, _, _, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_ce, test_kl, test_acc = evaluate_epoch(model, test_loader, device)
        if scheduler:
            scheduler.step()
            print(f"lr: {optimizer.param_groups[0]['lr']:.10f}")

        print(f"""epoch [{epoch+1}/{epochs}] train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
                    test loss: {test_loss:.3f} | test acc: {test_acc:.2f}%""")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        test_ces.append(test_ce)
        test_kls.append(test_kl)

    return train_losses, test_losses, test_ces, test_kls

def main() -> None:
    parser = argparse.ArgumentParser(description="training script with configurable hyperparameters.")
    parser.add_argument("--hidden1", type=int, default=300, help="size of first hidden layer")
    parser.add_argument("--hidden2", type=int, default=100, help="size of second hidden layer")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=bool, default=False, help="Enable learning rate decay")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    args = parser.parse_args()

    if not args.rnd_seed:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    params = MLPNetParams.from_args(args)
    print(params)

    dataset = MnistCsvDataset("data/mnist_data.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f"train set size: {train_size}, test set size: {test_size}")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, params.batch_size, shuffle=True)

    model = MLPNet(784, params.hidden1, params.hidden2, 10)
    print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))
    scheduler = None
    if params.lr_decay:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.94 ** (epoch // 2))

    train_losses, test_losses, test_i_x_t, test_i_t_y = train_model(
            model,
            train_loader, test_loader,
            optimizer, scheduler,
            params.device,
            params.epochs,
    )

    test_loss, _, _, test_acc = evaluate_epoch(model, test_loader, params.device)
    print(f"lr: {params.lr}, test loss: {test_loss}, test acc: {test_acc}")

    torch.save(model.state_dict(), f"{params.save_dir()}.pth")
    with open(f"{params.save_dir()}_stats.json", "w") as json_file:
        json.dump(params.to_json(test_losses, test_acc, list(zip(test_i_x_t, test_i_t_y))), json_file, indent=2)

    plot_losses(test_losses, train_losses, params.file_name(), params.save_dir())
    plot_information_plane(test_i_x_t, test_i_t_y, params.save_dir())

if __name__ == "__main__": main()
