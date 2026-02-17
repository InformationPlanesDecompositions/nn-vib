# just to play around and understand

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vib_mnist_train import VIBNet, evaluate_epoch, FashionMnistIdxDataset
from msc import get_device

h1 = 500
h2 = 300
z_dim = 125
o_shape = 10
i_shape = 784
path = f"save_stats_weights/vib_mnist_{h1}_{h2}_{z_dim}_0.2_0.0001/vib_mnist_{h1}_{h2}_{z_dim}_0.2_0.0001.pth"
weights = torch.load(path, map_location="cpu")
model = VIBNet(z_dim, 784, h1, h2, o_shape)
model.load_state_dict(weights)

print(model)

def plot_weight_heatmaps(model: nn.Module, cmap: str = "viridis") -> None:
  layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
  if not layers:
    return

  n = len(layers)
  cols = int(np.ceil(np.sqrt(n)))
  rows = int(np.ceil(n / cols))

  fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
  axes = np.array(axes).reshape(-1)

  for i, (name, module) in enumerate(layers):
    matrix = module.weight.detach().cpu().numpy()
    im = axes[i].imshow(matrix, aspect="auto", cmap=cmap)
    total_params = sum(p.numel() for p in module.parameters())
    axes[i].set_title(f"{name} ({total_params} params)")
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

  for j in range(n, len(axes)):
    axes[j].axis("off")

  plt.tight_layout()
  plt.show()

#plot_weight_heatmaps(model)

def magnitude_prune_top_percent(model: nn.Module, p: float) -> None:
  with torch.no_grad():
    for module in model.modules():
      if isinstance(module, nn.Linear):
        w = module.weight.data
        k = int(w.numel() * p)
        if k <= 0:
          continue
        idx = torch.topk(w.abs().view(-1), k, sorted=False).indices
        w.view(-1)[idx] = 0.0

device = get_device()
test_loader = DataLoader(
  FashionMnistIdxDataset("data/mnist_fashion/", train=False),
  batch_size=100,
  shuffle=False
)
beta = 0.2
base_loss, _, _, base_acc = evaluate_epoch(model.to(device), test_loader, device, beta=beta)
print(f"before prune: loss={base_loss:.6f}, acc={base_acc:.2f}")
for pct in [0.98, 0.96, 0.94]:
  pruned_model = VIBNet(z_dim, 784, h1, h2, o_shape).to(device)
  pruned_model.load_state_dict(weights)
  magnitude_prune_top_percent(pruned_model, pct)
  loss, _, _, acc = evaluate_epoch(pruned_model, test_loader, device, beta=beta)
  print(f"after prune {int(pct*100)}%: loss={loss:.6f}, acc={acc:.2f}")
