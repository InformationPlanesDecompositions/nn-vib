## VIB Research
```bash
uv sync
source .venv/bin/activate
```

- [ ] Train at more beta in the lower range
- [ ] Try training with dropout too?
- [ ] Does IB improve quantization of models because of its weight distribution change effect?
- [ ] Try training with beta scheduled so start at low compression and then to high or other way around

### Todo
- [ ] information plane during training save too
    - [ ] Look at actual I(Z;X) and I(Z;Y)
- [ ] data class for params to save more easily
- [ ] run hyper-parameter sweaps for all diff beta
- [ ] PCA of bottleneck layer (to see activity and clusters)
- [ ] plot based on losses and not really accuracies?
