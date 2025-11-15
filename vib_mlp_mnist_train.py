#!/usr/bin/env python3

import json
import os
import argparse
from typing import Tuple, List
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
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    self.labels = torch.tensor(data[:, 0], dtype=torch.long)
    self.images = torch.tensor(data[:, 1:], dtype=torch.float32)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx: int):
    return self.images[idx].view(1, 28, 28), self.labels[idx]

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
    self.fc2 = nn.Linear(hidden1, hidden2)

    self.fc_mu = nn.Linear(hidden2, z_dim)
    self.fc_sigma = nn.Linear(hidden2, z_dim)

    self.fc_decode = nn.Linear(z_dim, output_shape)

  def encode(self, x: torch.Tensor):
    h = F.relu(self.fc1(x))
    h = F.relu(self.fc2(h))
    mu = self.fc_mu(h)
    sigma = F.softplus(self.fc_sigma(h)) + 1e-8 #sigma = self.fc_sigma(h)
    return mu, sigma

  def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor):
    eps = torch.randn_like(sigma)
    return mu + sigma*eps

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_flat = x.view(x.size(0), -1)
    mu, sigma = self.encode(x_flat)
    z = self.reparameterize(mu, sigma)
    logits = self.fc_decode(z)
    return logits, mu, sigma

def vib_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    beta: float
) -> torch.Tensor:
  '''
  ce: lower bound on I(Z;Y) (prediction)
  kl: upper bound on I(Z;X) (compression)
  # (beta bigger = more compression)
  '''
  ce_loss = F.cross_entropy(logits, y)
  kl = 0.5*torch.sum(mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1, dim=1).mean()
  return ce_loss + beta*kl

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float
) -> Tuple[float, float]:
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X, Y in (tq := tqdm(dataloader, desc='training', leave=False)):
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()

    logits, mu, sigma = model(X)
    loss = vib_loss(logits, Y, mu, sigma, beta)
    loss.backward()
    optimizer.step()

    # accumulate (note multiply nll by batch size to match previous running_loss scheme)
    bs = X.size(0)
    running_loss += loss.item() * bs

    _, preds = torch.max(logits, 1)
    correct += (preds == Y).sum().item()
    total += bs

    tq.set_postfix({
      'loss': f'{loss.item():.4f}',
      'acc': f'{100.0 * correct / total:.2f}'
    })

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float
) -> Tuple[float, float]:
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      logits, mu, sigma = model(images)
      loss = vib_loss(logits, labels, mu, sigma, beta)

      bs = images.size(0)
      running_loss += loss.item() * bs

      _, preds = torch.max(logits, 1)
      correct += (preds == labels).sum().item()
      total += bs

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    beta: float,
) -> Tuple[List[float], List[float]]:
  model.to(device)
  train_losses, test_losses = [], []
  for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta=beta)
    test_loss, test_acc = evaluate(model, test_loader, device, beta=beta)
    print(f'''epoch [{epoch+1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
          \t    test loss: {test_loss:.3f} | test acc: {test_acc:.2f}%''')
    train_losses.append(train_loss)
    test_losses.append(test_loss)
  return train_losses, test_losses

def main() -> None:
  parser = argparse.ArgumentParser(description='Training script with configurable hyperparameters.')
  parser.add_argument('--beta', type=float, required=True, help='Beta coefficient')
  parser.add_argument('--z_dim', type=int, default=30, help='Latent dimension size (default: 30)')
  parser.add_argument('--hidden1', type=int, default=300, help='Size of first hidden layer (default: 300)')
  parser.add_argument('--hidden2', type=int, default=100, help='Size of second hidden layer (default: 100)')
  parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 500)')
  parser.add_argument('--rnd_seed', type=bool, default=False, help='Random torch seed or default of 42')
  parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-3)')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
  args = parser.parse_args()

  if not args.rnd_seed: torch.manual_seed(42)

  beta = args.beta
  z_dim = args.z_dim
  hidden1 = args.hidden1
  hidden2 = args.hidden2
  learning_rate = args.learning_rate
  epochs = args.epochs
  batch_size = args.batch_size
  device = get_device()

  save_dir = f'save_stats_weights/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}'
  os.makedirs(save_dir, exist_ok=True)

  print(
      f'hyperparameters:\n'
      f'  beta           = {beta}\n'
      f'  z_dim          = {z_dim}\n'
      f'  hidden1        = {hidden1}\n'
      f'  hidden2        = {hidden2}\n'
      f'  learning_rate  = {learning_rate}\n'
      f'  epochs         = {epochs}\n'
      f'  batch_size     = {batch_size}\n'
      f'  device         = {device}\n'
      f'  save_dir       = {save_dir}'
    )

  dataset = MnistCsvDataset('data/mnist_data.csv')
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  print(f'train set size: {train_size}, test set size: {test_size}')
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  model = VIBNet(z_dim, 784, hidden1, hidden2, 10)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, device, epochs, beta=beta)
  test_loss, test_acc = evaluate(model, test_loader, device, beta)
  print(f'test loss: {test_loss}, test acc: {test_acc}')

  torch.save(model.state_dict(), f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}.pth')

  with open(f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_stats.json', 'w') as json_file:
    json.dump({
      'beta': beta,
      'test_losses': test_losses,
      'test_acc': test_acc,
      'z_dim': z_dim,
      'hidden1': hidden1,
      'hidden2': hidden1,
      'learning_rate': learning_rate,
      'batch_size': batch_size,
      'epochs': epochs,
    }, json_file, indent=2)

  epochs = len(test_losses)
  plt.figure(figsize=(10, 6))
  plt.plot(range(1, epochs + 1), train_losses, marker='o', linewidth=2, markersize=6, color='#1f77b4', label='Training Loss')
  plt.plot(range(1, epochs + 1), test_losses, marker='o', linewidth=2, markersize=6, color='#ff7f0e', label='Test Loss')
  plt.title(f'(vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}) Loss', fontsize=16, fontweight='bold', pad=20)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Loss', fontsize=14)
  plt.grid(True, alpha=0.3)
  plt.legend(fontsize=12)
  plt.savefig(f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_test_loss.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
  main()
