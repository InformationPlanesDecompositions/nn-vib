import os, gzip, pickle, json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_device():
  device = ""
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.mps.is_available():
    device = "mps"
  else:
    device = "cpu"
  return torch.device(device)


@dataclass
class VIBNetParams:
  model_name: str
  beta: float
  z_dim: int
  hidden1: int
  hidden2: int
  decoder_hidden: int | None
  lr: float
  batch_size: int
  epochs: int
  device: torch.device
  rnd_seed: int

  @classmethod
  def from_args(cls, args: argparse.Namespace, model_name: str):
    return cls(
      model_name=model_name,
      beta=args.beta,
      z_dim=args.z_dim,
      hidden1=args.hidden1,
      hidden2=args.hidden2,
      decoder_hidden=getattr(args, "decoder_hidden", None),
      lr=args.lr,
      batch_size=args.batch_size,
      epochs=args.epochs,
      device=get_device(),
      rnd_seed=args.rnd_seed,
    )

  def file_name(self) -> str:
    if self.model_name == "cnn" and self.decoder_hidden is not None:
      return (
        f"vib_{self.model_name}_{self.hidden1}_{self.hidden2}_{self.decoder_hidden}_"
        f"{self.z_dim}_{self.beta}_{self.lr}_{self.epochs}_{self.rnd_seed}"
      )
    return f"vib_{self.model_name}_{self.hidden1}_{self.hidden2}_{self.z_dim}_{self.beta}_{self.lr}_{self.epochs}_{self.rnd_seed}"

  def save_dir(self) -> str:
    s = f"save_stats_weights/{self.file_name()}"
    os.makedirs(s, exist_ok=True)
    return f"{s}/{self.file_name()}"

  def to_json(self, train_loss, train_accuracy, test_losses, test_accuracy):
    return {
      "train_loss": train_loss,
      "train_acc": train_accuracy,
      "test_losses": test_losses,
      "test_acc": test_accuracy,
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
      f"\tmodel         = {self.model_name}\n"
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


def load_weights(filepath, verbose=True):
  weights = torch.load(filepath, map_location="cpu")
  if verbose:
    print(f"loaded object type: {type(weights)}")
  if isinstance(weights, dict):
    if verbose:
      print("keys in the weights file:")
    for key in weights.keys():
      if verbose:
        print(f"- {key} with shape {weights[key].shape}")

  return weights


class MnistCsvDataset(Dataset):
  def __init__(self, filepath: str):
    data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    self.labels = torch.tensor(data[:, 0], dtype=torch.long)
    self.images = torch.tensor(data[:, 1:], dtype=torch.float32)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx: int):
    return self.images[idx].view(1, 28, 28), self.labels[idx]


def idx_extractor(filepath: str) -> np.ndarray:
  open_func = gzip.open if filepath.endswith(".gz") else open
  with open_func(filepath, "rb") as f:
    magic = int.from_bytes(f.read(4), "big")
    num_dimensions = magic % 256
    dimensions = []
    for _ in range(num_dimensions):
      dimensions.append(int.from_bytes(f.read(4), "big"))
    data = np.frombuffer(f.read(), dtype=np.uint8)

  return data.reshape(dimensions)


class FashionMnistIdxDataset(Dataset):
  def __init__(self, data_dir: str, train: bool = True):
    prefix = "train" if train else "t10k"

    images_filepath = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte")
    labels_filepath = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte")

    if not os.path.exists(images_filepath) or not os.path.exists(labels_filepath):
      raise FileNotFoundError(
        f"Could not find required files in '{data_dir}'. "
        f"Expected: {os.path.basename(images_filepath)} and {os.path.basename(labels_filepath)}"
      )

    images_np = idx_extractor(images_filepath)
    labels_np = idx_extractor(labels_filepath)
    self.images = torch.from_numpy(images_np.copy()).float().div(255.0)
    self.labels = torch.from_numpy(labels_np.copy()).long()

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, idx: int):
    image = self.images[idx].view(1, 28, 28)
    label = self.labels[idx]
    return image, label


class CIFAR10Dataset(Dataset):
  """https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"""

  def __init__(self, data_dir: str, train: bool = True, transform=None):
    self.data_dir = data_dir
    self.transform = transform
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    image_batches, label_batches = [], []

    for batch_file in batch_files:
      batch_path = os.path.join(self.data_dir, batch_file)
      if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Could not find required file '{batch_file}' in '{self.data_dir}'")
      with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
      data = batch[b"data"] if b"data" in batch else batch["data"]
      labels = batch[b"labels"] if b"labels" in batch else batch["labels"]
      image_batches.append(data)
      label_batches.append(labels)

    images_np = np.vstack(image_batches).astype(np.uint8)
    labels_np = np.array(sum(label_batches, []), dtype=np.int64)

    # store as HWC so torchvision image transforms work naturally.
    self.images = images_np.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    self.labels = torch.from_numpy(labels_np.copy())

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, idx: int):
    image = self.images[idx]
    label = self.labels[idx]

    if self.transform is not None:
      image = self.transform(image)
    else:
      image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

    return image, label


def weights_location(h1, h2, z_dim, beta, lr):
  top_dir = "save_stats_weights"
  var = lambda v, w, x, y, z: f"vib_mnist_{v}_{w}_{x}_{y}_{z}"
  return f"{top_dir}/{var(h1, h2, z_dim, beta, lr)}/{var(h1, h2, z_dim, beta, lr)}.pth"


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


def evaluate_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  device: torch.device,
  beta: float,
  mc_samples: int = 10,
) -> Tuple[float, float, float, float]:
  model.eval()
  with torch.no_grad():
    total = 0
    total_ce = 0.0
    total_kl = 0.0
    total_correct = 0
    for X, Y in dataloader:
      X, Y = X.to(device), Y.to(device)
      bs = X.size(0)
      logits, mu, sigma = model(X)
      probs_sum = F.softmax(logits, dim=1)
      # monte carlo average over latent samples z for p(y|x), kept per-batch to avoid OOM.
      for _ in range(mc_samples - 1):
        logits, _, _ = model(X)
        probs_sum += F.softmax(logits, dim=1)
      probs = probs_sum / mc_samples
      ce = F.nll_loss(torch.log(probs.clamp_min(1e-8)), Y)
      variance = sigma.pow(2)
      log_variance = 2 * torch.log(sigma)
      kl_terms = 0.5 * (variance + mu.pow(2) - 1.0 - log_variance)
      kl = torch.sum(kl_terms, dim=1).mean()
      _, preds = torch.max(probs, 1)

      total += bs
      total_ce += ce.item() * bs
      total_kl += kl.item() * bs
      total_correct += (preds == Y).sum().item()

    avg_ce = total_ce / total
    avg_kl = total_kl / total
    loss = avg_ce + beta * avg_kl
    acc = 100.0 * total_correct / total
  return loss, avg_ce, avg_kl, acc
