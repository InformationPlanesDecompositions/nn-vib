from torchviz import make_dot
import torch

from vib_mlp_mnist_train import VIBNet
from msc import load_weights

h1, h2, z_dim, o_shape, beta = 300, 100, 30, 10, 0.001

model = VIBNet(z_dim, 784, h1, h2, o_shape)
weights = load_weights('save_stats_weights/vib_mnist_300_100_30_0.001/vib_mnist_300_100_30_0.001.pth', verbose=False)
model.load_state_dict(weights)

x = torch.randn(1, 28, 28)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render('plots/vibnet_computational_graph', format="png")
