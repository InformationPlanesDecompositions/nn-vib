import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import copy
from testing_utils import get_device

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
      input_shape: int=784,
      hidden1: int=300,
      hidden2: int=100,
      z_dim: int=100,
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
    eps = torch.randn_like(std)
    return mu + std*eps

  def forward(self, x):
    x_flat = x.view(x.size(0), -1)
    mu, std = self.encode(x_flat)
    z = self.reparameterize(mu, std)
    return self.decoder(z), mu, std

# kl-diverg?
def loss_function(y_pred, y, mu, std, beta):
  """
  y_pred : [batch_size,10]
  y : [batch_size,10]
  mu : [batch_size,z_dim]
  std: [batch_size,z_dim]
  """
  CE = F.cross_entropy(y_pred, y, reduction="sum")
  KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
  return (beta*KL + CE) / y.size(0)

def train_epoch(model, dataloader, optimizer, device, beta: float):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X, Y in (tq := tqdm(dataloader, desc="Training", leave=False)):
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()

    log_probs, mu, std = model(X)
    loss = loss_function(log_probs, Y, mu, std, beta)
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

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      log_probs, mu, std = model(images)
      loss = loss_function(log_probs, labels, mu, std, beta)

      bs = images.size(0)
      running_loss += loss.item() * bs

      _, preds = torch.max(log_probs, 1)
      correct += (preds == labels).sum().item()
      total += bs

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def train_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer,
    device,
    epochs: int,
    beta: float=1.0
):
  model.to(device)

  for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta=beta)

    test_loss, test_acc = evaluate(model, test_loader, device, beta=beta)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n")

if __name__ == "__main__":
  train_test_split = 0.8
  batch_size = 64
  z_dim = 100
  beta = 1e-3
  learning_rate = 1e-4
  epochs = 7

  dataset = MnistCsvDataset("../data/mnist_data.csv")
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f"train_size: {train_size}, test_size: {test_size}")

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  model = VIBLeNet(z_dim=z_dim)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  train_model(model, train_loader, test_loader, optimizer, device, epochs, beta=beta)

  torch.save(model.state_dict(), "../weights/vib_lenet_300_100_mnist.pth")

  # -------------

  prune_ps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
  original_model = copy.deepcopy(model)

  for prune_p in prune_ps:
    pruned_model = copy.deepcopy(original_model)
    prune.l1_unstructured(pruned_model.fc_mu, name="weight", amount=prune_p)
    test_loss, test_acc = evaluate(pruned_model, test_loader, device, beta=beta)
    print(f"Pruned: {prune_p*100:.2f} -> Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
