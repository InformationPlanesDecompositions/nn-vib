## Training Log
- FashionMNIST
- 4 different random seeds
- Random seeds: [42, 2136623168, 3824702233, 416282721, 3991408081]

- Analyze magnitude based pruning and SVD pruning
- Analyze with singular value decomposition (basis directions)
- Analyze neuron pruning
- Analyze heat maps of the weight matrices

### MLP IB
- Epochs 400
- Size 1: 386 -> 15 -> 128 [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
- Size 2: 256 -> 10 -> 64 [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
- Size 3: 512 -> 10 -> 128 [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
- Learning rate: 2e-4

### CNN IB
- Betas: [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
- Epochs 400
- Size 1:
- Size 2:
- Learning rate: 2e-4
