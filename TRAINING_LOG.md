# Training Log
```bash
source .venv/bin/activate
betas=(0.5 0.4 0.3 0.2 0.1 0.01 0.001 0.0001 0.0)
for beta in "${betas[@]}"; do
  python3 src/mlp/mlp_ib.py --beta "$beta" --rnd_seed 2136623168 --z_dim 10 --hidden1 256 --hidden2 64;
done
```

```bash
python3 src/mlp/inspect_mlp_ib.py \
  --save_root mlp_final_save_stats_weights \
  --prune_method incoming \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/mlp/inspect_mlp_ib.py \
  --save_root mlp_final_save_stats_weights \
  --prune_method outgoing \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/mlp/inspect_mlp_ib.py \
  --save_root mlp_final_save_stats_weights \
  --prune_method weight \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'
```

```bash
python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/mlp_pruning_report_incoming.json \
  --plots_dir plots/ \

python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/mlp_pruning_report_outgoing.json \
  --plots_dir plots/ \

python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/mlp_pruning_report_weight.json \
  --plots_dir plots/ \
```

## MLP IB
- FashionMNIST
- Random seeds: [2136623168, 3824702233, 416282721, 3991408081]
- Betas: [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
- Epochs: 300
- Learning rate: 2e-4

### 2136623168
#### Trained 1
- [X] Size 1: 256 -> 10 -> 64 (`./mlp_run.sh 2136623168 10 256 64`)
- [X] Size 2: 386 -> 15 -> 128 (`./mlp_run.sh 2136623168 15 386 128`)
- [X] Size 3: 512 -> 10 -> 128 (`./mlp_run.sh 2136623168 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 256 -> 4 -> 64 (`./mlp_run.sh 2136623168 4 256 64`)
- [X] Size 2: 386 -> 8 -> 128 (`./mlp_run.sh 2136623168 8 386 128`)
- [X] Size 3: 512 -> 4 -> 128 (`./mlp_run.sh 2136623168 4 512 128`)

### 3824702233
#### Trained 1
- [X] Size 1: 256 -> 10 -> 64 (`./mlp_run.sh 3824702233 10 256 64`)
- [X] Size 2: 386 -> 15 -> 128 (`./mlp_run.sh 3824702233 15 386 128`)
- [X] Size 3: 512 -> 10 -> 128 (`./mlp_run.sh 3824702233 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 256 -> 4 -> 64 (`./mlp_run.sh 3824702233 4 256 64`)
- [X] Size 2: 386 -> 8 -> 128 (`./mlp_run.sh 3824702233 8 386 128`)
- [X] Size 3: 512 -> 4 -> 128 (`./mlp_run.sh 3824702233 4 512 128`)

### 416282721
#### Trained 1
- [X] Size 1: 256 -> 10 -> 64 (`./mlp_run.sh 416282721 10 256 64`)
- [X] Size 2: 386 -> 15 -> 128 (`./mlp_run.sh 416282721 15 386 128`)
- [X] Size 3: 512 -> 10 -> 128 (`./mlp_run.sh 416282721 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 256 -> 4 -> 64 (`./mlp_run.sh 416282721 4 256 64`)
- [X] Size 2: 386 -> 8 -> 128 (`./mlp_run.sh 416282721 8 386 128`)
- [X] Size 3: 512 -> 4 -> 128 (`./mlp_run.sh 416282721 4 512 128`)

### 3991408081
#### Trained 1
- [X] Size 1: 256 -> 10 -> 64 (`./mlp_run.sh 3991408081 10 256 64`)
- [X] Size 2: 386 -> 15 -> 128 (`./mlp_run.sh 3991408081 15 386 128`)
- [X] Size 3: 512 -> 10 -> 128 (`./mlp_run.sh 3991408081 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 256 -> 4 -> 64 (`./mlp_run.sh 3991408081 4 256 64`)
- [X] Size 2: 386 -> 8 -> 128 (`./mlp_run.sh 3991408081 8 386 128`)
- [X] Size 3: 512 -> 4 -> 128 (`./mlp_run.sh 3991408081 4 512 128`)

---
---

## CNN IB
- CIFAR-10
- Random seeds: [2136623168, 3824702233, 416282721, 3991408081]
- Betas: [0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    -- TODO: POSSIBLY LOWER SETS OF BETAS --
- Epochs: 300
- Learning rate: 2e-4

### 2136623168
#### Trained 1
- [X] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 2136623168 20 6 16 84`)
- [X] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 2136623168 22 8 18 96`)
- [X] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 2136623168 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 2136623168 10 6 16 84`)
- [X] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 2136623168 12 8 18 96`)
- [X] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 2136623168 10 10 20 96`)

### 3824702233
#### Trained 1
- [X] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 3824702233 20 6 16 84`)
- [X] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 3824702233 22 8 18 96`)
- [X] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 3824702233 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 3824702233 10 6 16 84`)
- [X] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 3824702233 12 8 18 96`)
- [X] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 3824702233 10 10 20 96`)

# TODO: FORGOT TO TRAIN BETA=0.0 for all CNN's above --- ^ ---

### 416282721
#### Trained 1
- [ ] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 416282721 20 6 16 84`)
- [ ] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 416282721 22 8 18 96`)
- [ ] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 416282721 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 416282721 10 6 16 84`)
- [ ] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 416282721 12 8 18 96`)
- [ ] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 416282721 10 10 20 96`)

### 3991408081
#### Trained 1
- [ ] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 3991408081 20 6 16 84`)
- [ ] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 3991408081 22 8 18 96`)
- [ ] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 3991408081 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 3991408081 10 6 16 84`)
- [ ] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 3991408081 12 8 18 96`)
- [ ] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 3991408081 10 10 20 96`)
