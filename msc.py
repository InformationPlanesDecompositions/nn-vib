from typing import List, Optional
import matplotlib.pyplot as plt
import torch

def plot_x_y(
    xs: List,
    ys1: List,
    ys2: Optional[List],
    xlabel: str,
    ylabel: str,
    line_1_label: str,
    line_2_label: Optional[str],
    xlog: bool,
    point_labels: bool,
):
  fig, ax = plt.subplots(figsize=(10, 6))

  ax.plot(xs, ys1, marker='o', linestyle='-', color='b', label=line_1_label)
  if ys2 and line_2_label:
    ax.plot(xs, ys2, marker='x', linestyle='--', color='r', label=line_2_label)

  if xlog:
    ax.set_xscale('log')

  ax.set_title(f'{ylabel} vs {xlabel}')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  if point_labels:
    for x, y in zip(xs, ys1):
      ax.text(x, y, f'({x:.4f}, {y:.2f})', fontsize=8,
              verticalalignment='bottom', horizontalalignment='right')

    if ys2:
      for x, y in zip(xs, ys2):
        ax.text(x, y, f'({x:.4f}, {y:.2f})', fontsize=8,
                verticalalignment='top', horizontalalignment='left')

  ax.legend()
  return fig, ax

def get_device():
  device = ''
  if torch.cuda.is_available(): device = 'cuda'
  elif torch.mps.is_available(): device = 'mps'
  else: device = 'cpu'
  return torch.device(device)

def load_weights(filepath, verbose=True):
  weights = torch.load(filepath, map_location='cpu')
  if verbose: print(f"loaded object type: {type(weights)}")
  if isinstance(weights, dict):
    if verbose: print("keys in the weights file:")
    for key in weights.keys():
      if verbose: print(f"- {key} with shape {weights[key].shape}")

  return weights
