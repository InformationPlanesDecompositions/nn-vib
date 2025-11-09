import copy
import json
import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from inspect_weights import load_weights
from vib_lenet_300_100_mnist import (
    VIBLeNet, MnistCsvDataset, train_test_split,
    batch_size, z_dim, evaluate
)
from testing_utils import get_device
from msc.plotting import plot_x_y
#%matplotlib widget

torch.manual_seed(42)
device = get_device()
print(f'using device: {device}')

"""
def vib_prune_by_kl_contribution(model, data_loader, device, beta, keep_ratio=0.3):
  model.eval()
  kl_per_dim_list = []

  with torch.no_grad():
    for x, _ in data_loader:
      x = x.view(x.size(0), -1).to(device)
      mu, std = model.encode(x)
      std = std.clamp(min=1e-8)
      log_std = torch.log(std)
      kl_per_dim = 0.5 * (mu.pow(2) + std.pow(2) - 2*log_std - 1)
      kl_per_dim = kl_per_dim.mean(0)  # [z_dim]
      kl_per_dim_list.append(kl_per_dim.cpu())

  kl_scores = torch.stack(kl_per_dim_list).mean(0)  # [z_dim]
  z_dim = kl_scores.shape[0]

  k = max(1, int(z_dim * keep_ratio))
  _, topk_indices = torch.topk(kl_scores, k, largest=True)  # ← largest=True!

  mask = torch.zeros(z_dim, device=device)
  mask[topk_indices] = 1.0

  with torch.no_grad():
    # Zero ROWS in fc_mu/fc_std → [z_dim, hidden]
    model.fc_mu.weight.data *= mask[:, None]
    model.fc_mu.bias.data *= mask
    model.fc_std.weight.data *= mask[:, None]
    model.fc_std.bias.data *= mask

    # Zero COLUMNS in decoder → [10, z_dim]
    #model.decoder.weight.data *= mask[None, :]

  effective_z = int(mask.sum().item())

  return model, effective_z, kl_scores
"""

if __name__ == '__main__':
  model = VIBLeNet(z_dim=z_dim).to(device)
  weights = load_weights(f'../weights/vib_lenet_300_100_mnist.pth', verbose=False)
  model.load_state_dict(weights)

  dataset = MnistCsvDataset('../data/mnist_data.csv')
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f'train_size: {train_size}, test_size: {test_size}')

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  betas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

  prune_layers = [
      ('encoder.0', 0.30),
      ('encoder.2', 0.30),
      ('fc_mu', 0.40),
      ('fc_logvar', 0.40),
  ]

  og_acc_list, pruned_acc_list = [], []

  for b in tqdm(betas):
    pruned_model = copy.deepcopy(model)

    for layer_name, amount in prune_layers:
      module = dict(pruned_model.named_modules())[layer_name]
      prune.l1_unstructured(module, name='weight', amount=amount)

    _, pruned_test_acc = evaluate(pruned_model, test_loader, device, beta=b)
    _, og_test_acc = evaluate(model, test_loader, device, beta=b)

    og_acc_list.append(og_test_acc)
    pruned_acc_list.append(pruned_test_acc)

  for og, pruned, in zip(og_acc_list, pruned_acc_list):
    print(f'diff: {pruned-og:.3f} -> og: {og:.3f}, pruned: {pruned:.3f}')

  fig, ax = plot_x_y(betas, pruned_acc_list, og_acc_list, 'β', 'acc', 'pruned', 'no prune', True, False)
  fig.savefig('../plots/vib_lenet_300_100_mnist_beta_vs_pruned_acc.png', dpi=300, bbox_inches='tight')
