#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 <rnd_seed> <z_dim> <hidden1> <hidden2> <decoder_hidden>"
  exit 1
fi

rnd_seed="$1"
z_dim="$2"
hidden1="$3"
hidden2="$4"
decoder_hidden="$5"

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

source "$script_dir/.venv/bin/activate"

betas=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.0)

for beta in "${betas[@]}"; do
  python3 "$script_dir/src/cnn/cnn_ib.py" \
    --rnd_seed "$rnd_seed" \
    --beta "$beta" \
    --z_dim "$z_dim" \
    --hidden1 "$hidden1" \
    --hidden2 "$hidden2" \
    --decoder_hidden "$decoder_hidden"
done
