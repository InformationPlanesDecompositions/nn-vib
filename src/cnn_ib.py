#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from msc import VIBNetParams, CIFAR10Dataset, run_training_job


class VIBNet(nn.Module):
    def __init__(
        self,
        z_dim: int,
        input_shape: tuple[int, int, int],
        hidden1: int,
        hidden2: int,
        output_shape: int,
    ):
        super().__init__()

        channels, height, width = input_shape

        self.pool = nn.MaxPool2d(2, 2)
        pooled_height = height // 2
        pooled_width = width // 2
        if pooled_height == 0 or pooled_width == 0:
            raise ValueError("input_shape too small for pooling")
        self.flat_dim = hidden1 * pooled_height * pooled_width

        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, hidden2)
        self.fc_decode = nn.Linear(hidden2, output_shape)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor):
        h = F.relu(self.fc2(x))
        logits = self.fc_decode(h)
        return logits

    def forward(self, x: torch.Tensor):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        logits = self.decode(z)
        return logits, mu, sigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cnn vib training with configurable hyperparameters.")
    parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
    parser.add_argument("--z_dim", type=int, required=True, default=128, help="latent dimension size")
    parser.add_argument("--hidden1", type=int, required=True, default=96, help="number of conv channels")
    parser.add_argument("--hidden2", type=int, required=True, default=384, help="size of latent decoder layer")
    parser.add_argument("--epochs", type=int, required=True, default=150, help="number of training epochs")
    parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--data_dir", type=str, default="data/CIFAR-10/", help="dataset path")
    args = parser.parse_args()
    params = VIBNetParams.from_args(args, "cnn")
    model = VIBNet(params.z_dim, (3, 32, 32), params.hidden1, params.hidden2, 10)
    train_dataset = CIFAR10Dataset(args.data_dir, train=True)
    test_dataset = CIFAR10Dataset(args.data_dir, train=False)
    run_training_job(model, params, train_dataset, test_dataset)
