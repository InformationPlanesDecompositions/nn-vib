#!/usr/bin/env python3

import json
import os
import argparse
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

    self.input_shape = input_shape
    self.output_shape = output_shape

    self.fc1 = nn.Linear(input_shape, hidden1)
    self.fc2 = nn.Linear(hidden1, hidden2)
    self.fc_mu = nn.Linear(hidden2, z_dim)
    self.fc_logvar = nn.Linear(hidden2, z_dim)

    self.fc_decode = nn.Linear(z_dim, output_shape)

  def encode(self, x):
    h = F.relu(self.fc1(x))
    h = F.relu(self.fc2(h))
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    logvar = torch.clamp(logvar, min=-5, max=5)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5 * logvar) # std = sqrt(var)
      eps = torch.randn_like(std)
      return mu + eps * std
    return mu # deterministic at inference

  def forward(self, x):
    x_flat = x.view(x.size(0), -1)
    mu, logvar = self.encode(x_flat)
    z = self.reparameterize(mu, logvar)
    logits = self.fc_decode(z)
    return logits, mu, logvar

def vib_loss(logits, y, mu, logvar, beta):
  '''
  ce: lower bound on I(Z;Y) (prediction)
  kl: upper bound on I(Z;X) (compression)
  # beta bigger: more compression
  '''
  ce_loss = F.cross_entropy(logits, y, reduction='mean') # or reduct 'sum'
  kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
  return ce_loss + beta * kl

def train_epoch(model, dataloader, optimizer, device, beta: float):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X, Y in (tq := tqdm(dataloader, desc='training', leave=False)):
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()

    logits, mu, logvar = model(X)
    loss = vib_loss(logits, Y, mu, logvar, beta)
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

def evaluate(model, dataloader, device, beta: float):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      logits, mu, logvar = model(images)
      loss = vib_loss(logits, labels, mu, logvar, beta)

      bs = images.size(0)
      running_loss += loss.item() * bs

      _, preds = torch.max(logits, 1)
      correct += (preds == labels).sum().item()
      total += bs

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def train_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer,
    device,
    epochs: int,
    beta: float=1.0
):
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

def main():
  parser = argparse.ArgumentParser(description='Training script with configurable hyperparameters.')
  parser.add_argument('--beta', type=float, required=True, help='Beta coefficient')
  parser.add_argument('--z_dim', type=int, default=30, help='Latent dimension size (default: 30)')
  parser.add_argument('--hidden1', type=int, default=300, help='Size of first hidden layer (default: 300)')
  parser.add_argument('--hidden2', type=int, default=100, help='Size of second hidden layer (default: 100)')
  parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 500)')
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

  save_dir = f'save_stats_weights/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_{epochs}'
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

  torch.save(model.state_dict(), f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_{epochs}.pth')

  with open(f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_{epochs}_stats.json', 'w') as json_file:
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
  plt.title(f'(vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_{epochs}) Loss', fontsize=16, fontweight='bold', pad=20)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Loss', fontsize=14)
  plt.grid(True, alpha=0.3)
  plt.legend(fontsize=12)
  plt.savefig(f'{save_dir}/vib_mnist_{hidden1}_{hidden2}_{z_dim}_{beta}_{epochs}_test_loss.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
  main()
