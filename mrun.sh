#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <rnd_seed> <z_dim> <hidden1> <hidden2>"
    exit 1
fi

rnd_seed="$1"
z_dim="$2"
hidden1="$3"
hidden2="$4"

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# Activate the local virtual environment before running the sweep.
source "$script_dir/.venv/bin/activate"

betas=(0.5 0.4 0.3 0.2 0.1 0.01 0.001 0.0001)

for beta in "${betas[@]}"; do
    python3 "$script_dir/src/mlp/mlp_ib.py" \
        --beta "$beta" \
        --rnd_seed "$rnd_seed" \
        --z_dim "$z_dim" \
        --hidden1 "$hidden1" \
        --hidden2 "$hidden2"
done
