## Neural Network Information Research
- https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file

### Papers (comments means read)
- [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/pdf/1503.02406)
  - The general idea is to minimize the mutual information between X and Xhat while maximizing
    the mutual information between Xhat and Y
  - Basically the difficult problem with the IB method is that computing mutual information between
    two random variables is very difficult/expensive

- [Deep Variational Information Bottleneck](https://arxiv.org/pdf/1612.00410v7)

- [Compressing Neural Networks using the Variational Information Bottleneck](https://arxiv.org/pdf/1802.10399)

- [Nonlinear Information Bottleneck](https://arxiv.org/pdf/1705.02436)
  - Performs slightly better than VIB on MNIST/FasionMNIST
  - Much more clear separation in PCA in NIB than in VIB

### Notes
- Large weight values cause overly large and complex models prone to over fitting. This is why L1/L2
  regularization push the weights as close to 0 as possible to keep them sparse.
- Dropout is a similar form of regularization.

### Questions
- How can one measure generalization of a neural network and optimize on it? (implement as a cost function?)
- How can one measure the compression rate of a neural network?
- Measure the KL-Divergence between before and after pruning?
- How can one measure the agency of a neural network alone?
- A way to measure how well the models size fits to the problem? In a way so that you could bring the model
  down to a smallest possible size that fits more perfectly to the problem.
