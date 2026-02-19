import torch
import torch.nn.functional as F
from torch import nn

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
