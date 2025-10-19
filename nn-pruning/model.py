import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import numpy as np

class MnistCsvDataset(Dataset):
  def __init__(self, filepath):
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    self.labels = torch.tensor(data[:, 0], dtype=torch.long)
    self.images = torch.tensor(data[:, 1:], dtype=torch.float32)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.images[idx].view(1, 28, 28), self.labels[idx]

class SimpleMNIST(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)

  def forward(self, x):
    # flatten from (batch, 1, 28, 28) -> (batch, 784)
    x = x.view(x.size(0), -1)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
