#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os, json, argparse, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from msc import get_device, CIFAR10Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class VIBCNNParams:
    beta: float
    z_dim: int
    hidden1: int
    hidden2: int
    decoder_hidden: int
    lr: float
    batch_size: int
    epochs: int
    device: torch.device
    rnd_seed: int

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        device: torch.device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        return cls(
            beta=args.beta,
            z_dim=args.z_dim,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            decoder_hidden=args.decoder_hidden,
            lr=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            rnd_seed=args.rnd_seed,
        )

    def file_name(self) -> str:
        return (
            f"vib_cnn_{self.hidden1}_{self.hidden2}_{self.decoder_hidden}_"
            f"{self.z_dim}_{self.beta}_{self.lr}_{self.epochs}_{self.rnd_seed}"
        )

    def save_dir(self) -> str:
        s = f"../SaveStatsWeights/{self.file_name()}"
        os.makedirs(s, exist_ok=True)
        return f"{s}/{self.file_name()}"

    def to_json(self, train_ce_losses, train_accs, test_ce_losses, test_accs):
        return {
            "train_ce_losses": train_ce_losses,
            "train_accs": train_accs,
            "test_ce_losses": test_ce_losses,
            "test_accs": test_accs,
            "beta": self.beta,
            "z_dim": self.z_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "decoder_hidden": self.decoder_hidden,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "rnd_seed": self.rnd_seed,
        }

    def __str__(self):
        return (
            f"hyperparameters:\n"
            f"\tmodel         = cnn\n"
            f"\tbeta          = {self.beta}\n"
            f"\tz_dim         = {self.z_dim}\n"
            f"\thidden1       = {self.hidden1}\n"
            f"\thidden2       = {self.hidden2}\n"
            f"\tdecoder_hidden= {self.decoder_hidden}\n"
            f"\tlr            = {self.lr}\n"
            f"\tepochs        = {self.epochs}\n"
            f"\tbatch_size    = {self.batch_size}\n"
            f"\tdevice        = {self.device}\n"
            f"\trnd_seed      = {self.rnd_seed}\n"
            f"\tsave_dir      = {self.save_dir()}"
        )


class VIBNet(nn.Module):
    def __init__(
        self,
        z_dim: int,
        input_shape: tuple[int, int, int],
        hidden1: int,
        hidden2: int,
        decoder_hidden: int,
        output_shape: int,
    ):
        super().__init__()
        channels, _, _ = input_shape

        self.conv1 = nn.Conv2d(channels, hidden1, kernel_size=5)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self._forward_features(dummy).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc_mu_logvar = nn.Linear(120, 2 * z_dim)
        self.fc2 = nn.Linear(z_dim, decoder_hidden)
        self.fc_decode = nn.Linear(decoder_hidden, output_shape)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        return torch.flatten(x, 1)

    def encode(self, x: torch.Tensor):
        # both conv layer calls
        h = torch.tanh(self.fc1(self._forward_features(x)))
        mu, logvar = torch.chunk(self.fc_mu_logvar(h), 2, dim=1)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor):
        h = torch.tanh(self.fc2(x))
        logits = self.fc_decode(h)
        return logits

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma) if self.training else mu
        logits = self.decode(z)
        return logits, mu, sigma

    @staticmethod
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
    beta: float,
    batch_ce_losses: Optional[List[float]] = None,
) -> Tuple[float, float]:
    model.train()
    device = next(model.parameters()).device

    ce_sum, correct, num_examples = 0.0, 0, 0

    for X, Y in tqdm(dataloader, desc="training", leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        logits, mu, sigma = model(X)
        ce, _, loss = VIBNet.vib_loss(logits, Y, mu, sigma, beta)

        loss.backward()
        optimizer.step()

        if batch_ce_losses is not None:
            batch_ce_losses.append(ce.item())

        batch_size = Y.size(0)
        ce_sum += ce.item() * batch_size
        correct += (logits.argmax(dim=1) == Y).sum().item()
        num_examples += batch_size

    avg_ce_loss = ce_sum / num_examples
    accuracy = 100.0 * correct / num_examples

    return avg_ce_loss, accuracy


def evaluate_epoch(
    model: nn.Module, test_dataloader: DataLoader, beta: float
) -> Tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device

    ce_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, Y in test_dataloader:
            X, Y = X.to(device), Y.to(device)

            logits, mu, sigma = model(X)
            ce, _, _ = VIBNet.vib_loss(logits, Y, mu, sigma, beta)

            batch_size = X.size(0)
            ce_sum += ce.item() * batch_size
            correct += (logits.argmax(dim=1) == Y).sum().item()
            total += batch_size

    avg_ce_loss = ce_sum / total
    accuracy = 100.0 * correct / total
    return avg_ce_loss, accuracy


def print_epoch(epoch, epochs, beta, train_ce_loss, train_acc, test_ce_loss, test_acc):
    print(
        f"""epoch [{epoch + 1}/{epochs}] β({beta}) train ce loss: {train_ce_loss:.3f} | train acc: {train_acc:.2f}%
  \t\t\ttest ce loss: {test_ce_loss:.3f} | test acc: {test_acc:.2f}%"""
    )


def save_ce_losses_json(ce_losses: List[float], save_path: str) -> None:
    if not ce_losses:
        return

    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({"train_batch_ce_losses": ce_losses}, json_file, indent=2)


if __name__ == "__main__":
    epochs = 300
    batch_size = 128
    learning_rate = 3e-4

    parser = argparse.ArgumentParser(
        description="cnn vib training with configurable hyperparameters."
    )
    parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
    parser.add_argument(
        "--z_dim", type=int, required=True, help="latent dimension size"
    )
    parser.add_argument(
        "--hidden1",
        type=int,
        default=6,
        help="number of channels in the first conv layer",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=16,
        help="number of channels in the second conv layer",
    )
    parser.add_argument(
        "--decoder_hidden", type=int, default=84, help="number of post-ib hidden units"
    )
    parser.add_argument("--rnd_seed", type=int, required=True, help="random seed")
    parser.add_argument(
        "--data_dir", type=str, default="data/CIFAR-10/", help="dataset path"
    )
    parser.add_argument(
        "--plot_ce_losses",
        action="store_true",
        default=False,
        help="save per-batch training CE losses to JSON",
    )
    args = parser.parse_args()
    device = get_device()
    params = VIBCNNParams.from_args(args, device, epochs, batch_size, learning_rate)
    print(params)

    seed_everything(params.rnd_seed)

    train_transform = CIFAR10Dataset.train_transform(args.data_dir)
    test_transform = CIFAR10Dataset.test_transform(args.data_dir)

    num_workers = min(4, os.cpu_count() or 0)
    pin_memory = params.device.type == "cuda"
    train_loader_kwargs = {
        "batch_size": params.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(params.rnd_seed),
    }
    test_loader_kwargs = {
        "batch_size": params.batch_size * 4,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(params.rnd_seed),
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True

    train_dataset = CIFAR10Dataset(args.data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(args.data_dir, train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **test_loader_kwargs)

    model = VIBNet(
        params.z_dim,
        (3, 32, 32),
        params.hidden1,
        params.hidden2,
        params.decoder_hidden,
        10,
    ).to(device)
    print(
        f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))

    train_ce_losses, train_accs, test_ce_losses, test_accs = [], [], [], []
    batch_ce_losses = [] if args.plot_ce_losses else None

    for epoch in range(params.epochs):
        train_ce_loss, train_acc = train_epoch(
            model, train_loader, optimizer, beta=args.beta, batch_ce_losses=batch_ce_losses
        )

        if epoch % 10 == 0 and epoch != params.epochs - 1:
            test_ce_loss, test_acc = evaluate_epoch(model, test_loader, beta=args.beta)

            print_epoch(
                epoch,
                params.epochs,
                args.beta,
                train_ce_loss,
                train_acc,
                test_ce_loss,
                test_acc,
            )

            train_ce_losses.append(train_ce_loss)
            train_accs.append(train_acc)
            test_ce_losses.append(test_ce_loss)
            test_accs.append(test_acc)

    torch.save(model.state_dict(), f"{params.save_dir()}.pth")

    if batch_ce_losses is not None:
        losses_path = f"{params.save_dir()}_train_ce_losses.json"
        save_ce_losses_json(batch_ce_losses, losses_path)
        print(f"saved CE losses: {losses_path}")

    final_test_ce_loss, final_test_acc = evaluate_epoch(
        model, test_loader, beta=args.beta
    )
    test_ce_losses.append(final_test_ce_loss)
    test_accs.append(final_test_acc)

    with open(f"{params.save_dir()}_stats.json", "w") as json_file:
        json.dump(
            params.to_json(train_ce_losses, train_accs, test_ce_losses, test_accs),
            json_file,
            indent=2,
        )
