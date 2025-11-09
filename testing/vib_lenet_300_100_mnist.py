import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from testing_utils import get_device
from msc.plotting import plot_x_y

torch.manual_seed(42)
device = get_device()
print(f'using device: {device}')

class MnistCsvDataset(Dataset):
  def __init__(self, filepath: str):
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    self.labels = torch.tensor(data[:, 0], dtype=torch.long)
    self.images = torch.tensor(data[:, 1:], dtype=torch.float32)
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, idx: int):
    return self.images[idx].view(1, 28, 28), self.labels[idx]

class VIBLeNet(nn.Module):
  def __init__(
      self,
      z_dim: int,
      input_shape: int=784,
      hidden1: int=300,
      hidden2: int=100,
      output_shape: int=10
  ):
    super().__init__()

    self.input_shape = input_shape
    self.output_shape = output_shape

    self.encoder = nn.Sequential(
      nn.Linear(input_shape, hidden1),
      nn.ReLU(inplace=True),
      nn.Linear(hidden1, hidden2),
      nn.ReLU(inplace=True),
    )

    self.fc_mu = nn.Linear(hidden2, z_dim)
    self.fc_logvar = nn.Linear(hidden2, z_dim)

    self.decoder = nn.Linear(z_dim, output_shape)

  def reparameterize(self, mu, logvar):
    if mu.requires_grad:
      std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
      eps = torch.randn_like(std)
      return mu + eps * std
    else:
      return mu

  def forward(self, x):
    x_flat = x.view(x.size(0), -1)
    h = self.encoder(x_flat)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    z = self.reparameterize(mu, logvar)
    logits = self.decoder(z)
    return logits, mu, logvar

def vib_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float=1e-3,
) -> torch.Tensor:
  '''
  ce: lower bound on I(Z;Y) (prediction)
  kl: upper bound on I(Z;X) (compression)
  bigger beta = more compression
  '''

  ce = F.cross_entropy(logits, y, reduction='mean')

  logvar = torch.clamp(logvar, min=-10, max=10)
  kl_per_sample = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
  kl = kl_per_sample.mean()

  return ce + beta * kl

def train_epoch(model, dataloader, optimizer, device, beta: float):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for x, y in (t := tqdm(dataloader, desc='training', leave=False)):
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    logits, mu, logvar = model(x)
    loss = vib_loss(logits, y, mu, logvar, beta)
    loss.backward()
    optimizer.step()

    bs = x.size(0)
    running_loss += loss.item() * bs

    preds = logits.argmax(dim=1)
    correct += (preds == y).sum().item()
    total += bs

    t.set_postfix({
      'loss': f'{loss.item():.3f}',
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
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)
      logits, mu, logvar = model(x)
      loss = vib_loss(logits, y, mu, logvar, beta)

      bs = x.size(0)
      running_loss += loss.item() * bs

      preds = logits.argmax(dim=1)
      correct += (preds == y).sum().item()
      total += bs

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

z_dim = 16
base_beta = 1e-4
max_beta   = 2e-1

batch_size = 256
train_test_split = 0.8
learning_rate = 1e-3
weight_decay=1e-5
warmup_epochs = 18
epochs = 60

def curr_beta(epoch):
  if epoch < warmup_epochs:
    return base_beta + (max_beta - base_beta) * (epoch / warmup_epochs)
  else: return max_beta

if __name__ == '__main__':
  dataset = MnistCsvDataset('../data/mnist_data.csv')
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f'train_size: {train_size}, test_size: {test_size}')
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  model = VIBLeNet(z_dim=z_dim).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  test_stats = []
  for epoch in range(epochs):
    print(f'epoch {epoch}/{epochs}')
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, device, curr_beta(epoch)
    )
    test_loss, test_acc = evaluate(
        model, test_loader, device, curr_beta(epoch)
    )
    print(f'train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%')
    print(f'test  loss: {test_loss:.4f} | test  acc: {test_acc:.2f}%\n')
    test_stats.append((epoch, curr_beta(epoch), test_loss, test_acc))

    scheduler.step()

  torch.save(model.state_dict(), f'../weights/vib_lenet_300_100_mnist.pth')

  for (epoch, beta, tloss, tacc) in test_stats:
    print(f'epoch {epoch}: beta {beta:.3f}, loss {tloss:.3f}, acc {tacc:.3f}')

  #avg_loss, acc = evaluate(model, test_loader, device, max_beta)
  #fig, ax = plot_x_y(betas, acc_list, None, 'β', 'acc', 'accuracy', None, False, True)
  #fig.savefig('../plots/vib_lenet_300_100_mnist_beta_vs_acc.png', dpi=300, bbox_inches='tight')
