#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <beta_value>"
    echo "Example: $0 0.1"
    exit 1
fi

beta=$1
learning_rates=(0.05 0.01 0.005 0.001 0.0005 0.0001)
max_jobs=2

echo "--- starting training runs for beta = ${beta} ---"

for lr in "${learning_rates[@]}"; do
    echo "submitting job for lr = ${lr}"

    ./src/vib_mlp_mnist_train.py --beta "${beta}" --lr "${lr}" &

  # wait for at least one background job to finish if the limit is reached
  while (( $(jobs -r | wc -l) >= max_jobs )); do
      wait -n
  done
done

echo "--- waiting for remaining jobs to complete... ---"
wait

echo "--- all jobs finished. ---"
