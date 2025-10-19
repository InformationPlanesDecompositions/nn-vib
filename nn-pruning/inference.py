import torch
from model import SimpleMNIST

if __name__ == "__main__":
  model = SimpleMNIST()
  model.load_state_dict(torch.load("weights/simple_mnist_mlp.pth"))
  model.eval()
