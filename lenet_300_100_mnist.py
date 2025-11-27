import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from msc import get_device

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

class LeNet(nn.Module):
    def __init__(self, input_size: int=784, hidden1: int=300, hidden2: int=100, num_classes: int=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_decode = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1) # 28x28 to 784
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_decode(x) # (32,10)
        x = F.log_softmax(x, dim=1)
        return x

def train_epoch(model, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in (tq := tqdm(dataloader, desc='training', leave=False)):
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
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}'
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

def train_model(model, train_loader, test_loader: DataLoader, criterion, optimizer, scheduler, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        print(f'epoch [{epoch+1}/{epochs}]')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f'train loss: {train_loss:.4f} | train acc: {train_acc:.2f}%')
        print(f'test    loss: {test_loss:.4f} | test    acc: {test_acc:.2f}%\n')

if __name__ == '__main__':
    dataset = MnistCsvDataset('data/mnist_data.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f'train_size: {train_size}, test_size: {test_size}')

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = LeNet()
    epochs = 200
    learning_rate = 1e-4
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97** (epoch // 2))

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs)

    torch.save(model.state_dict(), 'save_stats_weights/vib_mnist_300_100_default_novib/vib_mnist_300_100_default_novib.pth')
