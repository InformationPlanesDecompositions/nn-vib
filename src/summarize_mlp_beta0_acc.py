#!/usr/bin/env python3
import json, os, re, sys
from collections import defaultdict

run_name_pattern = re.compile(
  r"^vib_mlp_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$"
)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} <save_root>")

  save_root = sys.argv[1]
  grouped = defaultdict(list)

  for run_name in sorted(os.listdir(save_root)):
    run_dir = os.path.join(save_root, run_name)
    if not os.path.isdir(run_dir):
      continue

    match = run_name_pattern.match(run_name)
    if not match:
      continue

    hidden1_s, hidden2_s, z_dim_s, beta_s, lr_s, epochs_s, seed_s = match.groups()
    if float(beta_s) != 0.0:
      continue

    stats_path = os.path.join(run_dir, f"{run_name}_stats.json")
    if not os.path.isfile(stats_path):
      continue

    with open(stats_path, "r", encoding="utf-8") as f:
      stats = json.load(f)

    key = (int(hidden1_s), int(hidden2_s), int(z_dim_s), float(lr_s), int(epochs_s))
    grouped[key].append((int(seed_s), float(stats["test_accs"][-1])))

  print("hidden1,hidden2,z_dim,lr,epochs,num_seeds,avg_final_test_acc,seeds")
  for key in sorted(grouped):
    values = sorted(grouped[key])
    avg_acc = sum(acc for _, acc in values) / len(values)
    seeds = " ".join(str(seed) for seed, _ in values)
    hidden1, hidden2, z_dim, lr, epochs = key
    print(f"{hidden1},{hidden2},{z_dim},{lr:g},{epochs},{len(values)},{avg_acc:.6f},{seeds}")
