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
- [ ] increase model size to over parameterize
- [ ] inspect better generalization for pruned over parameterized IB networks
- [ ] plot original model as well (ie. beta == 0.0)
