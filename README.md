## Neural Network Information Bottleneck Research

### Papers (comments means read)
- [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/pdf/1503.02406)
  - The general idea is to minimize the mutual information between X and Xhat while maximizing
    the mutual information between Xhat and Y
  - Basically the difficult problem with the IB method is that computing mutual information between
    two random variables is very difficult/expensive

- [Deep Variational Information Bottleneck](https://arxiv.org/pdf/1612.00410v7)
  - ce and beta * kl

- [Compressing Neural Networks using the Variational Information Bottleneck](https://arxiv.org/pdf/1802.10399)
  - Sota method basically because they compared to all other current methods and basically beat
    them on compression and accuracy (and the trade off)

- [Nonlinear Information Bottleneck](https://arxiv.org/pdf/1705.02436)
  - Performs slightly better than VIB on MNIST/FasionMNIST
  - Much more clear separation in PCA in NIB than in VIB
  - But needs O(n^2) compared to VIB O(n)

### Notes
- Large weight values cause overly large and complex models prone to over fitting. This is why L1/L2
  regularization push the weights as close to 0 as possible to keep them sparse.

### Questions
- How can one measure generalization of a neural network and optimize on it? (implement as a cost function?)
  - Most likely just compression
- Is the pruneability a good measure for over-parameterization of a neural network?
  - Yes.
- What happens if I let it train to try and reach some sort of "grokking" with an ib layer?
- How do the weights change even if you just keep training for a while? (could test this by
  simply training on 200+ epoch)

### Data
- https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file
