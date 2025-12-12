## vib
```bash
uv sync
source .venv/bin/activate
```

### next steps
- [ ] standard over same beta/lr and diff epochs to inspect grokking
- [ ] plot based on losses, not just accuracy
- [ ] fix information plane running ce, kl plots
- [ ] log scale y-axis on weight distributions
- [ ] why for diff beta are they not effected the same in order by pruning?
    - (implying some beta better than others for use cases)
- [ ] increase model size to over parameterize (500 -> 125 -> 300)
- [ ] inspect better generalization for pruned over parameterized IB networks
- [ ] plot original model as well (ie. beta == 0.0)
- [ ] Switch to using Fashion-MNIST

- [ ] something with pausing training in the encoder only at some point,
    but keep learning in decoder (think this was for v-information though)
