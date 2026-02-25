#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from msc import VIBNetParams, CIFAR10Dataset, run_training_job, FashionMnistIdxDataset


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
        x_flat = x.view(x.size(0), -1)
        mu, sigma = self.encode(x_flat)
        z = self.reparameterize(mu, sigma)
        logits = self.decode(z)
        return logits, mu, sigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mlp vib training with configurable hyperparameters.")
    parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
    parser.add_argument("--z_dim", type=int, required=True, default=128, help="latent dimension size")
    parser.add_argument("--hidden1", type=int, required=True, default=1024, help="size of first hidden layer")
    parser.add_argument("--hidden2", type=int, required=True, default=512, help="size of second hidden layer")
    parser.add_argument("--epochs", type=int, required=True, default=200, help="number of training epochs")
    parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--data_dir", type=str, default="data/CIFAR-10/", help="dataset path")
    args = parser.parse_args()
    params = VIBNetParams.from_args(args, "mlp")
    # train_dataset = CIFAR10Dataset(args.data_dir, train=True)
    # test_dataset = CIFAR10Dataset(args.data_dir, train=False)
    train_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=True)
    test_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=False)
    model = VIBNet(params.z_dim, 784, params.hidden1, params.hidden2, 10)
    run_training_job(model, params, train_dataset, test_dataset)
