import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from data import MnistCsvDataset
from msc import get_device
from typing import List

torch.manual_seed(42)
device = get_device()
print(f'using device: {device}')

class VIBLeNet(nn.Module):
  def __init__(
      self,
      z_dim: int,
      input_shape: int=784,
      hidden1: int=1024,
      hidden2: int=1024,
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
    x = self.encoder(x)
    return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)

  def reparameterize(self, mu, std):
    '''
    mu: mean u
    std: standard deviation sigma
    '''
    eps = torch.randn_like(std)
    return mu + std*eps

  def forward(self, x):
    x_flat = x.view(x.size(0), -1)
    mu, std = self.encode(x_flat)
    z = self.reparameterize(mu, std)
    return self.decoder(z), mu, std

def loss_function(y_pred, y, mu, std, beta):
  '''
  y_pred : [batch_size,10]
  y : [batch_size,10]
  mu : [batch_size,z_dim]
  std: [batch_size,z_dim]

  CE: lower bound on I(Z;Y) (prediction)
  KL: upper bound on I(Z;X) (compression)
  # beta bigger: more compression
  '''
  std = std.clamp(min=1e-8)
  CE = F.cross_entropy(y_pred, y, reduction='sum')
  KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
  return (beta*KL + CE) / y.size(0)

def train_epoch(model, dataloader, optimizer, device, beta: float):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for X, Y in (tq := tqdm(dataloader, desc='training', leave=False)):
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
      'loss': f'{loss.item():.4f}',
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
    scheduler,
    device,
    epochs: int,
    beta: float=1.0
):
  model.to(device)
  test_losses = []
  for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, beta=beta)
    test_loss, test_acc = evaluate(model, test_loader, device, beta=beta)
    print(f'''epoch [{epoch+1}/{epochs}] β({beta}) train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%
          \t    test loss: {test_loss:.3f} | test acc: {test_acc:.2f}%''')
    test_losses.append(test_loss)
    scheduler.step()
  return test_losses

def print_layer_boxplot_stats(
    model: torch.nn.Module,
    layer_names: List[str],
    include_mean: bool=True,
    decimals: int=4,
) -> None:
    weight_values = []
    format_str = f'{{:.{decimals}f}}'

    print('=== Box Plot Statistics per Layer ===\n')
    for layer_name in layer_names:
        module = dict(model.named_modules()).get(layer_name)
        if module is not None and hasattr(module, 'weight') and module.weight is not None:
            weights = module.weight.data.flatten().cpu().numpy()
            weight_values.append(weights)

            # Compute stats
            min_val = np.min(weights)
            q1 = np.percentile(weights, 25)
            median = np.median(weights)
            q3 = np.percentile(weights, 75)
            max_val = np.max(weights)
            mean_val = np.mean(weights) if include_mean else None

            print(f'Layer: {layer_name}')
            print(f'  Min:     {format_str.format(min_val)}')
            print(f'  Q1:      {format_str.format(q1)}')
            print(f'  Median:  {format_str.format(median)}')
            print(f'  Q3:      {format_str.format(q3)}')
            print(f'  Max:     {format_str.format(max_val)}')
            if include_mean: print(f'  Mean:    {format_str.format(mean_val)}')
            print('-' * 40)
        else:
            print(f'Layer: {layer_name} — [No weight found or layer missing]')
            print('-' * 40)

    if weight_values:
        all_weights = np.concatenate(weight_values)

        min_val = np.min(all_weights)
        q1 = np.percentile(all_weights, 25)
        median = np.median(all_weights)
        q3 = np.percentile(all_weights, 75)
        max_val = np.max(all_weights)
        mean_val = np.mean(all_weights) if include_mean else None

        print('=== Combined All Layers ===')
        print(f'  Min:     {format_str.format(min_val)}')
        print(f'  Q1:      {format_str.format(q1)}')
        print(f'  Median:  {format_str.format(median)}')
        print(f'  Q3:      {format_str.format(q3)}')
        print(f'  Max:     {format_str.format(max_val)}')
        if include_mean:
            print(f'  Mean:    {format_str.format(mean_val)}')
    else:
        print('No weights found in any of the specified layers.')


# === Example Usage ===
# layer_names = ['encoder.0', 'encoder.2', 'fc_mu', 'fc_std']
# print_layer_boxplot_stats(model, layer_names)

beta = 0.3
z_dim = 30
hidden1 = 300
hidden2 = 100
train_test_split = 0.8
batch_size = 128
learning_rate = 1e-2
epochs = 500

if __name__ == '__main__':
  dataset = MnistCsvDataset('data/mnist_data.csv')
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f'train_size: {train_size}, test_size: {test_size}')
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  model = VIBLeNet(hidden1=hidden1, hidden2=hidden2, z_dim=z_dim)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/5), gamma=0.3)

  test_losses = train_model(model, train_loader, test_loader, optimizer, scheduler, device, epochs, beta=beta)
  test_loss, test_acc = evaluate(model, test_loader, device, beta)
  torch.save(model.state_dict(), f'weights/vib_mnist_{hidden1}_{hidden2}_mnist_{beta}_{epochs}.pth')

  print(f'avg test loss: {test_loss}, avg test acc: {test_acc}')

  with open('weights/vib_mlp_mnist_stats.json', 'w') as json_file:
    json.dump({
      'beta': beta,
      'test_losses': test_losses,
      'test_acc': test_acc,
      'hidden1': hidden1,
      'hidden2': hidden1,
      'z_dim': z_dim,
      'learning_rate': learning_rate,
      'batch_size': batch_size,
      'epochs': epochs,
      'train_test_split': train_test_split,
    }, json_file, indent=2)

  epochs = len(test_losses)
  plt.figure(figsize=(10, 6))
  plt.plot(range(1, epochs + 1), test_losses, marker='o', linewidth=2, markersize=6, color='#1f77b4')
  plt.title(f'Test Loss over Epochs (β = {beta})', fontsize=16, fontweight='bold', pad=20)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Test Loss', fontsize=14)
  plt.grid(True, alpha=0.3)

  # Annotate final test loss
  final_loss = test_losses[-1]
  plt.annotate(f'{final_loss:.3f}',
               xy=(epochs, final_loss),
               xytext=(10, 0),
               textcoords='offset points',
               fontsize=12,
               color='darkred',
               fontweight='bold')

  plt.tight_layout()
  plt.savefig(f'plots/vib_mlp_mnist_loss_{beta}_{epochs}.png', dpi=300, bbox_inches='tight')

  print_layer_boxplot_stats(model, ['encoder.0', 'encoder.2', 'fc_mu', 'fc_std'])
