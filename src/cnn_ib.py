#!/usr/bin/env python3
import argparse
import torch
from torch import nn, optim
from torchvision import transforms
from msc import VIBNetParams, CIFAR10Dataset, run_training_job


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

    self.encoder = nn.Sequential(
      nn.Conv2d(channels, hidden1, kernel_size=3, padding=1),
      nn.BatchNorm2d(hidden1),
      nn.ReLU(inplace=True),
      nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1),
      nn.BatchNorm2d(hidden1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1),
      nn.BatchNorm2d(hidden2),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(1),
    )
    self.flat_dim = hidden2

    self.fc_mu = nn.Linear(self.flat_dim, z_dim)
    self.fc_logvar = nn.Linear(self.flat_dim, z_dim)
    self.fc1 = nn.Linear(z_dim, decoder_hidden)
    self.fc_decode = nn.Linear(decoder_hidden, output_shape)

  def encode(self, x: torch.Tensor):
    h = self.encoder(x)
    h = torch.flatten(h, 1)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    logvar = torch.clamp(logvar, min=-10.0, max=2.0)
    sigma = torch.exp(0.5 * logvar)
    return mu, sigma

  def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, x: torch.Tensor):
    h = torch.relu(self.fc1(x))
    return self.fc_decode(h)

  def forward(self, x: torch.Tensor):
    mu, sigma = self.encode(x)
    z = self.reparameterize(mu, sigma)
    logits = self.decode(z)
    return logits, mu, sigma


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="cnn vib training with configurable hyperparameters.")
  parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
  parser.add_argument("--z_dim", type=int, required=True, default=128, help="latent dimension size")
  parser.add_argument("--hidden1", type=int, required=True, default=96, help="number of channels in the first conv block")
  parser.add_argument("--hidden2", type=int, required=True, default=128, help="number of channels in the deeper conv blocks")
  parser.add_argument("--decoder_hidden", type=int, default=64, help="number of post-ib hidden units")
  parser.add_argument("--epochs", type=int, required=True, default=150, help="number of training epochs")
  parser.add_argument("--rnd_seed", type=int, default=42, help="torch manual seed (default: 42)")
  parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size")
  parser.add_argument("--data_dir", type=str, default="data/CIFAR-10/", help="dataset path")
  args = parser.parse_args()

  mean = (0.4914, 0.4822, 0.4465)
  std = (0.2023, 0.1994, 0.2010)

  train_transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.Normalize(mean, std),
    ]
  )

  test_transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ]
  )

  params = VIBNetParams.from_args(args, "cnn")
  model = VIBNet(params.z_dim, (3, 32, 32), params.hidden1, params.hidden2, args.decoder_hidden, 10)
  optimizer = optim.Adam(model.parameters(), lr=params.lr)
  train_dataset = CIFAR10Dataset(args.data_dir, train=True, transform=train_transform)
  test_dataset = CIFAR10Dataset(args.data_dir, train=False, transform=test_transform)
  run_training_job(model, optimizer, params, train_dataset, test_dataset)
