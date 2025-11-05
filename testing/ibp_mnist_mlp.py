import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from tqdm import tqdm

"""
goal:
  - train mnist using the information bottleneck principle loss function
"""

def get_device():
  device = ""
  if torch.cuda.is_available(): device = "cuda"
  elif torch.mps.is_available(): device = "mps"
  else: device = "cpu"
  print(f"Using device: {device}")
  return torch.device(device)

device = get_device()
torch.manual_seed(42)

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

def train_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for images, labels in (tq := tqdm(dataloader, desc="Training", leave=False)):
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)
    _, preds = torch.max(outputs, 1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

    tq.set_postfix({
      "loss": f"{loss.item():.4f}",
      "acc": f"{100.0 * correct / total:.2f}"
    })

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * images.size(0)
      _, preds = torch.max(outputs, 1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)

  avg_loss = running_loss / total
  accuracy = 100.0 * correct / total
  return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
  model.to(device)

  for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n")

if __name__ == "__main__":
  dataset = MnistCsvDataset("../data/mnist_data.csv")
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  print(f"train_size: {train_size}, test_size: {test_size}")

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  model = SimpleMNIST()
  lr = 1e-3
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5)

  torch.save(model.state_dict(), "../weights/simple_mnist_mlp.pth")
