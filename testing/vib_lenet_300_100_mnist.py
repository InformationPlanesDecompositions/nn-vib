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
print(f"using device: {device}")

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
    self.fc_std = nn.Linear(hidden2, z_dim)

    self.decoder = nn.Linear(z_dim, output_shape)

  def encode(self, x):
    """
    x: [batch_size,784]
    """
    x = self.encoder(x)
    return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)

  def reparameterize(self, mu, std):
    """
    mu: mean u
    std: standard deviation sigma
    """
    eps = torch.randn_like(std)
    return mu + std*eps

  def forward(self, x):
    x_flat = x.view(x.size(0), -1)
    mu, std = self.encode(x_flat)
    z = self.reparameterize(mu, std)
    return self.decoder(z), mu, std

def loss_function(y_pred, y, mu, std, beta, eps=1e-8):
  """
  CE: lower bound on I(Z;Y) (prediction)
  KL: upper bound on I(Z;X) (compression)
  bigger beta = more compression
  """
  CE = F.cross_entropy(y_pred, y, reduction="sum")
  KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * (std + eps).log() - 1)
  return (CE + beta*KL) / y.size(0), KL, CE

def train_epoch(model, dataloader, optimizer, device, beta: float):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X, Y in (tq := tqdm(dataloader, desc="training", leave=False)):
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()

    log_probs, mu, std = model(X)
    loss, KL, CE = loss_function(log_probs, Y, mu, std, beta)
    loss.backward()
    optimizer.step()

    # accumulate (note multiply nll by batch size to match previous running_loss scheme)
    bs = X.size(0)
    running_loss += loss.item() * bs

    _, preds = torch.max(log_probs, 1)
    correct += (preds == Y).sum().item()
    total += bs

    tq.set_postfix({
      "loss": f"{loss.item():.4f}",
      "acc": f"{100.0 * correct / total:.2f}"
    })

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def evaluate(model, dataloader, device, beta: float):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0
  running_KL = 0.0
  running_CE = 0.0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      log_probs, mu, std = model(images)
      loss, KL, CE = loss_function(log_probs, labels, mu, std, beta)

      bs = images.size(0)
      running_loss += loss.item() * bs

      _, preds = torch.max(log_probs, 1)
      correct += (preds == labels).sum().item()
      total += bs
      running_KL += KL
      running_CE += CE

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  average_KL = running_KL / total
  average_CE = running_CE / total
  return avg_loss, accuracy, average_KL, average_CE

def train_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer,
    scheduler,
    device,
    epochs: int,
    beta_func
):
  model.to(device)
  test_stats = []
  for epoch in range(epochs):
    print(f"epoch [{epoch+1}/{epochs}], beta: {beta_func(epoch):.3f}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta_func(epoch))
    test_loss, test_acc, avg_KL, avg_CE = evaluate(model, test_loader, device, beta_func(epoch))

    print(f"train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%")
    print(f"test  loss: {test_loss:.4f} | test  acc: {test_acc:.2f}%\n")

    test_stats.append((epoch, beta_func(epoch), test_loss, test_acc))

    scheduler.step()
  return test_stats

z_dim = 10
base_beta = 1e-4
max_beta   = 2e-1
batch_size = 256
train_test_split = 0.8
learning_rate = 1e-3
weight_decay=1e-5
warmup_epochs = 18
epochs = 60

def get_current_beta(epoch):
  if epoch < warmup_epochs:
    return base_beta + (max_beta - base_beta) * (epoch / warmup_epochs)
  else: return max_beta

if __name__ == "__main__":
  dataset = MnistCsvDataset("../data/mnist_data.csv")
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f"train_size: {train_size}, test_size: {test_size}")

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  model = VIBLeNet(z_dim=z_dim)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  test_stats = train_model(
      model,
      train_loader,
      test_loader,
      optimizer,
      scheduler,
      device,
      epochs,
      get_current_beta
  )

  torch.save(model.state_dict(), f"../weights/vib_lenet_300_100_mnist.pth")

  for (epoch, beta, tloss, tacc) in test_stats:
    print(f'epoch {epoch}: beta {beta:.3f}, loss {tloss:.3f}, acc {tacc:.3f}')

  #avg_loss, acc = evaluate(model, test_loader, device, max_beta)

  #fig, ax = plot_x_y(betas, acc_list, None, 'β', 'acc', 'accuracy', None, False, True)
  #fig.savefig('../plots/vib_lenet_300_100_mnist_beta_vs_acc.png', dpi=300, bbox_inches='tight')

  #with open('../weights/accuracy_data.json', 'w') as json_file:
  #  json.dump({'betas': betas, 'acc_list': acc_list}, json_file, indent=2)
