import copy
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import random_split, DataLoader
from inspect_weights import load_weights
from vib_lenet_300_100_mnist import (
    VIBLeNet, MnistCsvDataset, train_test_split,
    batch_size, beta, z_dim, evaluate
)
from testing_utils import get_device

torch.manual_seed(42)
device = get_device()
print(f"using device: {device}")

if __name__ == "__main__":
  model = VIBLeNet(z_dim=z_dim).to(device)
  weights = load_weights("../weights/vib_lenet_300_100_mnist.pth")
  model.load_state_dict(weights)

  dataset = MnistCsvDataset("../data/mnist_data.csv")
  train_size = int(train_test_split * len(dataset))
  test_size = len(dataset) - train_size
  print(f"train_size: {train_size}, test_size: {test_size}")

  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

  train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

  prune_ps = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
  original_model = copy.deepcopy(model)

  weight_layers = [
      ("encoder", "0", "weight"),
      ("encoder", "2", "weight"),
      ("decoder", "weight"),
  ]

  test_loss, test_acc = evaluate(original_model, test_loader, device, beta=beta)
  print(f"base -> test loss: {test_loss:.4f} | test acc: {test_acc:.2f}%")

  print("\npruning only fc_mu.weight")
  for prune_p in prune_ps:
    pruned_model = copy.deepcopy(original_model)
    prune.l1_unstructured(pruned_model.fc_mu, name="weight", amount=prune_p)
    test_loss, test_acc = evaluate(pruned_model, test_loader, device, beta=beta)
    print(f"pruned: {prune_p*100:.1f}% -> test loss: {test_loss:.4f} | test acc: {test_acc:.2f}%")

  print("\nper weight layer percent prune")
  for prune_p in prune_ps:
      pruned_model = copy.deepcopy(original_model)
      for layer in weight_layers:
        param_name = layer[-1]
        module_path = ".".join(layer[:-1])

        module = pruned_model
        for part in module_path.split("."): module = getattr(module, part)

        prune.l1_unstructured(module, name=param_name, amount=prune_p)

      test_loss, test_acc = evaluate(pruned_model, test_loader, device, beta=beta)
      print(f"pruned: {prune_p*100:.1f}% -> test loss: {test_loss:.4f} | test acc: {test_acc:.2f}%")
