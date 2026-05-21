# Training Log
```bash
python3 src/mlp/inspect_mlp_ib.py \
  --save_root ../bachelor-arbeit/Results/mlp_final_save_stats_weights \
  --prune_method incoming \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/mlp/inspect_mlp_ib.py \
  --save_root ../bachelor-arbeit/Results/mlp_final_save_stats_weights \
  --prune_method outgoing \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/mlp/inspect_mlp_ib.py \
  --save_root ../bachelor-arbeit/Results/mlp_final_save_stats_weights \
  --prune_method weight \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'
```

```bash
python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/inspect-saves/mlp_pruning_report_incoming.json \
  --plots_dir plots/ \
  --metric acc

python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/inspect-saves/mlp_pruning_report_outgoing.json \
  --plots_dir plots/ \
  --metric acc

python3 src/mlp/plot_mlp_pruning_robustness.py \
  --input_json plots/inspect-saves/mlp_pruning_report_weight.json \
  --plots_dir plots/ \
  --metric acc
```

## MLP IB
- FashionMNIST
- Random seeds: [2136623168, 3824702233, 416282721, 3991408081]
- Betas=(0.5 0.4 0.3 0.2 0.1 0.01 0.001 0.0001 0.0)
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

```
python3 src/cnn/inspect_cnn_ib.py \
  --save_root ../bachelor-arbeit/Results/cnn_save_stats_weights \
  --prune_method incoming \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/cnn/inspect_cnn_ib.py \
  --save_root ../bachelor-arbeit/Results/cnn_save_stats_weights \
  --prune_method outgoing \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'

python3 src/cnn/inspect_cnn_ib.py \
  --save_root ../bachelor-arbeit/Results/cnn_save_stats_weights \
  --prune_method weight \
  --layer_sets '[["fc_mu", "fc_logvar", "fc2"], ["fc2"]]'
```

```
python3 src/cnn/plot_cnn_pruning_robustness.py \
  --input_json plots/inspect-saves/cnn_pruning_report_incoming.json \
  --plots_dir plots/ \
  --metric acc

python3 src/cnn/plot_cnn_pruning_robustness.py \
  --input_json plots/inspect-saves/cnn_pruning_report_outgoing.json \
  --plots_dir plots/ \
  --metric acc

python3 src/cnn/plot_cnn_pruning_robustness.py \
  --input_json plots/inspect-saves/cnn_pruning_report_weight.json \
  --plots_dir plots/ \
  --metric acc
```

## CNN IB
- CIFAR-10
- Random seeds: [2136623168, 3824702233, 416282721, 3991408081]
- Betas=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.0)
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

### 416282721
#### Trained 1
- [X] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 416282721 20 6 16 84`)
- [X] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 416282721 22 8 18 96`)
- [X] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 416282721 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 416282721 10 6 16 84`)
- [X] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 416282721 12 8 18 96`)
- [X] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 416282721 10 10 20 96`)

### 3991408081
#### Trained 1
- [X] Size 1: 6 -> 16 -> (20) -> 84 (`./cnn_run.sh 3991408081 20 6 16 84`)
- [X] Size 2: 8 -> 18 -> (22) -> 96 (`./cnn_run.sh 3991408081 22 8 18 96`)
- [X] Size 3: 10 -> 20 -> (20) -> 96 (`./cnn_run.sh 3991408081 20 10 20 96`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 6 -> 16 -> (10) -> 84 (`./cnn_run.sh 3991408081 10 6 16 84`)
- [X] Size 2: 8 -> 18 -> (12) -> 96 (`./cnn_run.sh 3991408081 12 8 18 96`)
- [X] Size 3: 10 -> 20 -> (10) -> 96 (`./cnn_run.sh 3991408081 10 10 20 96`)
