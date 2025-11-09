import torch

def load_weights(filepath, verbose=True):
  weights = torch.load(filepath, map_location='cpu')
  if verbose: print(f"loaded object type: {type(weights)}")
  if isinstance(weights, dict):
    if verbose: print("keys in the weights file:")
    for key in weights.keys():
      if verbose: print(f"- {key} with shape {weights[key].shape}")

  return weights

if __name__ == "__main__":
  torch.set_printoptions(profile="full")

  weights = load_weights("../weights/lenet_300_100_mnist.pth")
  #print(weights)

  #fc_mu_weights = weights["fc_mu.weight"]
  #fc_std_weights = weights["fc_std.weight"]

  #print(f"fc mu weights: {fc_mu_weights}")
  #print(f"fc std weights: {fc_std_weights}")
