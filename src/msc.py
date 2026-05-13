import os, gzip, pickle, json
import argparse
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def get_device():
  device = ""
  if torch.cuda.is_available(): device = "cuda"
  elif torch.mps.is_available(): device = "mps"
  else: device = "cpu"
  return torch.device(device)
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
