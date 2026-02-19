import torch
import torch.nn.functional as F
from torch import nn

class VIBNet(nn.Module):
  def __init__(
    self,
    z_dim: int,
    input_shape: tuple[int, int, int],
    hidden1: int,
    hidden2: int,
    output_shape: int,
  ):
    super().__init__()
    channels, height, width = input_shape
    self.conv1 = nn.Conv2d(channels, hidden1, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    pooled_height = height // 2
    pooled_width = width // 2
    if pooled_height == 0 or pooled_width == 0:
      raise ValueError("input_shape too small for pooling")
    self.flat_dim = hidden1 * pooled_height * pooled_width
    self.fc_mu = nn.Linear(self.flat_dim, z_dim)
    self.fc_logvar = nn.Linear(self.flat_dim, z_dim)
    self.fc2 = nn.Linear(z_dim, hidden2)
    self.fc_decode = nn.Linear(hidden2, output_shape)

  def encode(self, x: torch.Tensor):
    h = F.relu(self.conv1(x))
    h = self.pool(h)
    h = h.view(h.size(0), -1)
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
    mu, sigma = self.encode(x)
    z = self.reparameterize(mu, sigma)
    logits = self.decode(z)
    return logits, mu, sigma
