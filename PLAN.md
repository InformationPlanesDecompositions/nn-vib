## Plan

### Todo/Plan Information Bottleneck Pruning
- [ ] What does pruning look like in an IB CNN?
- [ ] Train IB MLP and IB CNN
  - 3 different sizes with 5 different random seeds
  - FashionMNIST
  - Analyze magnitude based pruning and SVD pruning
  - Analyze with singular value decomposition (basis directions)
  - Look at much smaller `z_dim`
- [X] Look at pruning whole neurons?
- [X] Heat map of weight matrices to actually see what SVD base directions are indicating
- [ ] Make ib layer one matrix?

### Questions/Suspicions
- It seems as though everything after the bottleneck layer including the bottleneck layer (except
  the final decode layer) is much more sparse and pruneable?
- How does the model know to basically skip using all other weights and sort of thread a needle through
  only 4 specific neurons?
- Any current research on measuring over-parameterization?
