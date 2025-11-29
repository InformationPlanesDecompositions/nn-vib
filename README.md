## VIB Research
- Large weight values cause overly large and complex models prone to over fitting. This is why L1/L2
    regularization push the weights as close to 0 as possible to keep them sparse.
- Need to improve generalization
- Can the information bottleneck be used to understand generalization much better?
- https://www.youtube.com/watch?v=AKMuA_TVz3A&t=2207s
    - Distribution matching: |C(concat(X,Y))| < |C(X)| + |C(Y)| + O(1)
    - Compression == prediction
    - Kolmogorov complexity is the ultimate compressor

- How can one measure generalization of a neural network and optimize on it? (implement as a cost function?)
    - Most likely just compression
- Is the pruneability a good measure for over-parameterization of a neural network?
    - Yes.
- What happens if I let it train to try and reach some sort of "grokking" with an ib layer?
- How do the weights change even if you just keep training for a while? (could test this by
    simply training on 200+ epoch)
- Does the mutual information between layer give us some way to measurably choose a better beta

- [ ] Train at more beta in the lower range
- [ ] Train at higher beta just to see (play with learning rates and epochs)
- [ ] Look at actual I(Z;X) and I(Z;Y)
- [ ] Try training with dropout too?

- [ ] Inspect the layer after IB pruneability on different/bigger networks and datasets
