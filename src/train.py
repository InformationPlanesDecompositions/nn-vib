#!/usr/bin/env python3
import json
import argparse
from typing import Tuple, List
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from msc import (
  plot_information_plane,
  plot_losses,
  FashionMnistIdxDataset,
  VIBNetParams,
)
from mlp_ib import VIBNet as MlpVIBNet
from cnn_ib import VIBNet as CnnVIBNet

def vib_loss(
  logits: torch.Tensor,
  y: torch.Tensor,
  mu: torch.Tensor,
  sigma: torch.Tensor,
  beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  kl: upper bound on I(Z;X) (compression)
  ce: proxy for H(Y|Z) (relevance term)
  (beta bigger = more compression)
  """
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
  device: torch.device,
  beta: float,
) -> Tuple[float, float]:
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0
  for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    logits, mu, sigma = model(X)
    _, _, loss = vib_loss(logits, Y, mu, sigma, beta)
    loss.backward()
    optimizer.step()
    bs = X.size(0)
    running_loss += loss.item() * bs
    _, preds = torch.max(logits, 1)
    correct += (preds == Y).sum().item()
    total += bs
    tq.set_postfix(
      {
        "loss": f"{loss.item():.4f}",
        "acc": f"{100.0 * correct / total:.2f}",
      }
    )
  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def evaluate_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  device: torch.device,
  beta: float,
) -> Tuple[float, float, float, float]:
  model.eval()
  with torch.no_grad():
    batches = list(dataloader)
    Xs, Ys = zip(*batches)
    X = torch.cat(Xs, dim=0).to(device)
    Y = torch.cat(Ys, dim=0).to(device)
    logits, mu, sigma = model(X)
    ce, kl, loss = vib_loss(logits, Y, mu, sigma, beta)
    _, preds = torch.max(logits, 1)
    acc = 100.0 * (preds == Y).float().mean().item()
  return loss.item(), ce.item(), kl.item(), acc

def train_model(
  model: nn.Module,
  train_loader: DataLoader,
  test_loader: DataLoader,
  optimizer: optim.Optimizer,
  device: torch.device,
  epochs: int,
  beta: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
  model.to(device)
  train_losses, test_losses, test_ces, test_kls = [], [], [], []
  for epoch in range(epochs):
    train_loss, train_acc = train_epoch(
      model,
      train_loader,
      optimizer,
      device,
      beta=beta,
    )
    test_loss, ce, kl, test_acc = evaluate_epoch(
      model,
      test_loader,
      device,
      beta=beta,
    )
    print(
      f"""epoch [{epoch + 1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
    \t\t\ttest loss: {test_loss:.3f} | test acc: {test_acc:.2f}%"""
    )
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_ces.append(ce)
    test_kls.append(kl)
  return train_losses, test_losses, test_ces, test_kls

def build_model(model_name: str, params: VIBNetParams):
  if model_name == "mlp":
    return MlpVIBNet(
      params.z_dim,
      784,
      params.hidden1,
      params.hidden2,
      10,
    )
  return CnnVIBNet(
    params.z_dim,
    (1, 28, 28),
    params.hidden1,
    params.hidden2,
    10,
  )

def main() -> None:
  parser = argparse.ArgumentParser(description="training script with configurable hyperparameters.")
  parser.add_argument("--model", type=str, required=True, default="mlp", choices=["mlp", "cnn"], help="model type")
  parser.add_argument("--beta", type=float, required=True, help="beta coefficient")
  parser.add_argument("--z_dim", type=int, required=True, default=125, help="latent dimension size")
  parser.add_argument("--hidden1", type=int, required=True, default=500, help="size of first hidden layer")
  parser.add_argument("--hidden2", type=int, required=True, default=300, help="size of second hidden layer")
  parser.add_argument("--epochs", type=int, required=True, default=200, help="number of training epochs")
  parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
  parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
  parser.add_argument("--batch_size", type=int, default=128, help="batch size")
  parser.add_argument("--data_dir", type=str, default="data/mnist_fashion/", help="dataset path")
  args = parser.parse_args()
  if not args.rnd_seed:
    torch.manual_seed(42)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(42)
  params = VIBNetParams.from_args(args, args.model)
  print(params)
  train_dataset = FashionMnistIdxDataset(args.data_dir, train=True)
  test_dataset = FashionMnistIdxDataset(args.data_dir, train=False)
  train_loader = DataLoader(train_dataset, params.batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, params.batch_size, shuffle=True)
  model = build_model(args.model, params)
  print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
  optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.5, 0.999))
  train_losses, test_losses, test_ces, test_kls = train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    params.device,
    params.epochs,
    beta=params.beta,
  )
  test_loss, _, _, test_acc = evaluate_epoch(model, test_loader, params.device, params.beta)
  print(f"lr: {params.lr}, test loss: {test_loss}, test acc: {test_acc}")
  torch.save(model.state_dict(), f"{params.save_dir()}.pth")
  with open(f"{params.save_dir()}_stats.json", "w") as json_file:
    json.dump(
      params.to_json(test_losses, test_acc, list(zip(test_ces, test_kls))),
      json_file,
      indent=2,
    )
  plot_losses(test_losses, train_losses, params.file_name(), params.save_dir())
  plot_information_plane(test_ces, test_kls, params.save_dir())

if __name__ == "__main__":
  main()
