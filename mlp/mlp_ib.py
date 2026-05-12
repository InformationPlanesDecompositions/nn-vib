#!/usr/bin/env python3
from typing import Tuple
import os, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from msc import VIBNetParams, get_device, FashionMnistIdxDataset, vib_loss
from tqdm import tqdm

device = get_device()

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

def train_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  optimizer: optim.Optimizer,
  device: torch.device,
  beta: float,
) -> Tuple[float, float]:
  model.train()

  loss_sum, correct, num_examples = 0.0, 0, 0

  for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()

    logits, mu, sigma = model(X)
    ce, _, loss = vib_loss(logits, Y, mu, sigma, beta)

    loss.backward()
    optimizer.step()

    batch_size = Y.size(0)
    loss_sum += ce * batch_size
    correct += (logits.argmax(dim=1) == Y).sum().item()
    num_examples += batch_size

  avg_loss = loss_sum / num_examples
  accuracy = 100.0 * correct / num_examples

  return avg_loss, accuracy

def evaluate_epoch(model: nn.Module, test_dataloader: DataLoader, beta: float) -> Tuple[float, float]:
  model.eval()

  loss_sum = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for X, Y in test_dataloader:
      X, Y = X.to(device), Y.to(device)

      logits, mu, sigma = model(X)
      _, _, loss = vib_loss(logits, Y, mu, sigma, beta)

      batch_size = X.size(0)
      loss_sum += loss.item() * batch_size
      correct += (logits.argmax(dim=1) == Y).sum().item()
      total += batch_size

  avg_loss = loss_sum / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def print_epoch(epoch, epochs, beta, train_loss, train_acc, test_loss, test_acc):
  print(
    f"""epoch [{epoch + 1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
  \t\t\ttest loss: {test_loss:.3f} | test acc: {test_acc:.2f}%"""
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="mlp vib training with configurable hyperparameters.")
  parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
  parser.add_argument("--z_dim", type=int, required=True, help="latent dimension size")
  parser.add_argument("--hidden1", type=int, required=True, help="size of first hidden layer")
  parser.add_argument("--hidden2", type=int, required=True, help="size of second hidden layer")
  parser.add_argument("--epochs", type=int, required=True, help="number of training epochs")
  parser.add_argument("--rnd_seed", type=int, default=42, help="torch manual seed (default: 42)")
  parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size")
  args = parser.parse_args()
  params = VIBNetParams.from_args(args, "mlp")
  print(params)

  torch.manual_seed(params.rnd_seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed(params.rnd_seed)

  # ----
  num_workers = min(4, os.cpu_count() or 0)
  pin_memory = params.device.type == "cuda"
  loader_kwargs = {
    "batch_size": params.batch_size,
    "num_workers": num_workers,
    "pin_memory": pin_memory,
  }
  if num_workers > 0: loader_kwargs["persistent_workers"] = True

  train_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=True)
  test_dataset = FashionMnistIdxDataset("data/mnist_fashion/", train=False)
  train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
  test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
  # ----

  model = VIBNet(params.z_dim, 784, params.hidden1, params.hidden2, 10).to(device)
  print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
  optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))

  train_losses, train_accs, test_losses, test_accs = [], [], [], []

  for epoch in range(args.epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta=args.beta)

    if epoch % 10 == 0:
      test_loss, test_acc = evaluate_epoch(model, test_loader, beta=args.beta)

      print_epoch(epoch, args.epochs, args.beta, train_loss, train_acc, test_loss, test_acc)

      train_losses.append(train_loss); train_accs.append(train_acc)
      test_losses.append(test_loss); test_accs.append(test_acc)

  torch.save(model.state_dict(), f"{params.save_dir()}.pth")

  with open(f"{params.save_dir()}_stats.json", "w") as json_file:
    json.dump(
      params.to_json(train_losses[-1], train_accs[-1], test_losses, test_accs[-1]),
      json_file,
      indent=2,
    )
