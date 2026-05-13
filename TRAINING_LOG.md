# Training Log
```bash
source .venv/bin/activate
betas=(0.5 0.4 0.3 0.2 0.1 0.01 0.001 0.0001)
for beta in "${betas[@]}"; do
  python3 src/mlp/mlp_ib.py --beta "$beta" --rnd_seed 2136623168 --z_dim 10 --hidden1 256 --hidden2 64;
done
```

## MLP IB
- FashionMNIST
- Random seeds: [42, 2136623168, 3824702233, 416282721, 3991408081]
- Betas: [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
- Epochs: 300
- Learning rate: 2e-4

### 2136623168
#### Trained 1
- [X] Size 1: 256 -> 10 -> 64 (`./mrun.sh 2136623168 10 256 64`)
- [X] Size 2: 386 -> 15 -> 128 (`./mrun.sh 2136623168 15 386 128`)
- [X] Size 3: 512 -> 10 -> 128 (`./mrun.sh 2136623168 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [X] Size 1: 256 -> 4 -> 64 (`./mrun.sh 2136623168 4 256 64`)
- [X] Size 2: 386 -> 8 -> 128 (`./mrun.sh 2136623168 8 386 128`)
- [X] Size 3: 512 -> 4 -> 128 (`./mrun.sh 2136623168 4 512 128`)

### 3824702233
#### Trained 1
- [ ] Size 1: 256 -> 10 -> 64 (`./mrun.sh 3824702233 10 256 64`)
- [ ] Size 2: 386 -> 15 -> 128 (`./mrun.sh 3824702233 15 386 128`)
- [ ] Size 3: 512 -> 10 -> 128 (`./mrun.sh 3824702233 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1: 256 -> 4 -> 64 (`./mrun.sh 3824702233 4 256 64`)
- [ ] Size 2: 386 -> 8 -> 128 (`./mrun.sh 3824702233 8 386 128`)
- [ ] Size 3: 512 -> 4 -> 128 (`./mrun.sh 3824702233 4 512 128`)

### 416282721
#### Trained 1
- [ ] Size 1: 256 -> 10 -> 64 (`./mrun.sh 416282721 10 256 64`)
- [ ] Size 2: 386 -> 15 -> 128 (`./mrun.sh 416282721 15 386 128`)
- [ ] Size 3: 512 -> 10 -> 128 (`./mrun.sh 416282721 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1: 256 -> 4 -> 64 (`./mrun.sh 416282721 4 256 64`)
- [ ] Size 2: 386 -> 8 -> 128 (`./mrun.sh 416282721 8 386 128`)
- [ ] Size 3: 512 -> 4 -> 128 (`./mrun.sh 416282721 4 512 128`)

### 3991408081
#### Trained 1
- [ ] Size 1: 256 -> 10 -> 64 (`./mrun.sh 3991408081 10 256 64`)
- [ ] Size 2: 386 -> 15 -> 128 (`./mrun.sh 3991408081 15 386 128`)
- [ ] Size 3: 512 -> 10 -> 128 (`./mrun.sh 3991408081 10 512 128`)

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1: 256 -> 4 -> 64 (`./mrun.sh 3991408081 4 256 64`)
- [ ] Size 2: 386 -> 8 -> 128 (`./mrun.sh 3991408081 8 386 128`)
- [ ] Size 3: 512 -> 4 -> 128 (`./mrun.sh 3991408081 4 512 128`)

---
---

## CNN IB
- Betas: [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.0]
- Epochs 200
- Learning rate: 2e-4

### Notes

### 2136623168
#### Trained 1
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

### 3824702233
#### Trained 1
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

### 416282721
#### Trained 1
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

### 3991408081
#### Trained 1
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:

#### Trained 2 (smaller bottleneck layers)
- [ ] Size 1:
- [ ] Size 2:
- [ ] Size 3:
