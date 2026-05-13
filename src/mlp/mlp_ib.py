#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple
import os, json, argparse, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from msc import get_device, FashionMnistIdxDataset

# rnd_seeds: [2136623168, 3824702233, 416282721, 3991408081]
# betas:     [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
# epochs:    300
# lr:        2e-4
# models:
#   512 -> 10 -> 128
#   386 -> 15 -> 128
#   256 -> 10 -> 64
#   512 -> 4 -> 128
#   386 -> 8 -> 128
#   256 -> 4 -> 64

@dataclass
class VIBMLPParams:
  beta: float
  z_dim: int
  hidden1: int
  hidden2: int
  lr: float
  batch_size: int
  epochs: int
  device: torch.device
  rnd_seed: int

  @classmethod
  def from_args(cls, args: argparse.Namespace, device: torch.device):
    return cls(
      beta=args.beta,
      z_dim=args.z_dim,
      hidden1=args.hidden1,
      hidden2=args.hidden2,
      lr=args.lr,
      batch_size=args.batch_size,
      epochs=args.epochs,
      device=device,
      rnd_seed=args.rnd_seed,
    )

  def file_name(self) -> str:
    return f"vib_mlp_{self.hidden1}_{self.hidden2}_{self.z_dim}_{self.beta}_{self.lr}_{self.epochs}_{self.rnd_seed}"

  def save_dir(self) -> str:
    s = f"save_stats_weights/{self.file_name()}"
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
      "lr": self.lr,
      "batch_size": self.batch_size,
      "epochs": self.epochs,
      "rnd_seed": self.rnd_seed,
    }

  def __str__(self):
    return (
      f"hyperparameters:\n"
      f"\tmodel         = mlp\n"
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

class VIBMLP(nn.Module):
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

  @staticmethod
  def vib_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    beta: float,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, y)

    # KL(q(z|x) || p(z)) for diagonal Gaussians with p(z)=N(0, I).
    variance = sigma.pow(2)
    log_variance = 2 * torch.log(sigma)
    kl_terms = 0.5 * (variance + mu.pow(2) - 1.0 - log_variance)
    kl = torch.sum(kl_terms, dim=1).mean()

    # classification fit + compressed latent regularization.
    total_loss = ce + beta * kl
    return ce, kl, total_loss

def train_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  optimizer: optim.Optimizer,
  beta: float,
) -> Tuple[float, float]:
  model.train()
  device = next(model.parameters()).device

  ce_sum, correct, num_examples = 0.0, 0, 0

  for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()

    logits, mu, sigma = model(X)
    ce, _, loss = VIBMLP.vib_loss(logits, Y, mu, sigma, beta)

    loss.backward()
    optimizer.step()

    batch_size = Y.size(0)
    ce_sum += ce.item() * batch_size
    correct += (logits.argmax(dim=1) == Y).sum().item()
    num_examples += batch_size

  avg_ce_loss = ce_sum / num_examples
  accuracy = 100.0 * correct / num_examples

  return avg_ce_loss, accuracy

def evaluate_epoch(model: nn.Module, test_dataloader: DataLoader, beta: float) -> Tuple[float, float]:
  model.eval()
  device = next(model.parameters()).device

  ce_sum = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for X, Y in test_dataloader:
      X, Y = X.to(device), Y.to(device)

      logits, mu, sigma = model(X)
      ce, _, _ = VIBMLP.vib_loss(logits, Y, mu, sigma, beta)

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="mlp vib training with configurable hyperparameters.")
  parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
  parser.add_argument("--z_dim", type=int, required=True, help="latent dimension size")
  parser.add_argument("--hidden1", type=int, required=True, help="size of first hidden layer")
  parser.add_argument("--hidden2", type=int, required=True, help="size of second hidden layer")
  parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
  parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
  parser.add_argument("--rnd_seed", type=int, default=42, help="torch manual seed (default: 42)")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size")
  args = parser.parse_args()
  device = get_device()
  params = VIBMLPParams.from_args(args, device)
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

  model = VIBMLP(params.z_dim, 784, params.hidden1, params.hidden2, 10).to(device)
  print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
  optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))

  train_ce_losses, train_accs, test_ce_losses, test_accs = [], [], [], []

  for epoch in range(args.epochs):
    train_ce_loss, train_acc = train_epoch(model, train_loader, optimizer, beta=args.beta)

    if epoch % 10 == 0 and epoch != args.epochs - 1:
      test_ce_loss, test_acc = evaluate_epoch(model, test_loader, beta=args.beta)

      print_epoch(epoch, args.epochs, args.beta, train_ce_loss, train_acc, test_ce_loss, test_acc)

      train_ce_losses.append(train_ce_loss); train_accs.append(train_acc)
      test_ce_losses.append(test_ce_loss); test_accs.append(test_acc)

  torch.save(model.state_dict(), f"{params.save_dir()}.pth")

  final_test_ce_loss, final_test_acc = evaluate_epoch(model, test_loader, beta=args.beta)
  test_ce_losses.append(final_test_ce_loss)
  test_accs.append(final_test_acc)

  with open(f"{params.save_dir()}_stats.json", "w") as json_file:
    json.dump(
      params.to_json(train_ce_losses, train_accs, test_ce_losses, test_accs),
      json_file,
      indent=2,
    )
