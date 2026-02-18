## Plan

### Todo/Plan Information Bottleneck Pruning
- [ ] What does pruning look like in an IB CNN?
- [ ] Train IB MLP and IB CNN
  - 3 different sizes with 5 different random seeds
  - FashionMNIST
  - Analyze magnitude based pruning and SVD pruning
  - Analyze with singular value decomposition (basis directions)
  - Look at much smaller `z_dim`
- [ ] Look at pruning whole neurons?
- [ ] Heat map of weight matrices to actually see what SVD base directions are indicating
- [ ] Make ib layer one matrix?

### New testing infra:
- `src/mlp_ib.py`, `src/cnn_ib.py`:
  - holding models with parameters for input/output/internal layer size
  - basically like this:
    ```python
    @dataclass
    class VIBNetParams:
      beta: float
      z_dim: int
      hidden1: int
      hidden2: int

      lr: float
      lr_decay: bool
      batch_size: int
      epochs: int

      device: torch.device
      rnd_seed: bool
    ```
    but for the cnn I guess make the `hidden1` actually just conv layers?
- `src/train.py`: choose between either model and be able to pass in all parameters via cli

### Questions/Suspicions
- It seems as though everything after the bottleneck layer including the bottleneck layer (except
  the final decode layer) is much more sparse and pruneable?
- How does the model know to basically skip using all other weights and sort of thread a needle through
  only 4 specific neurons?
