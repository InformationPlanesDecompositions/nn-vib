#!/bin/bash

# TODO: add a start and stop date/time

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: $0 <num_jobs> <beta_value>"
    echo "example (2 parallel jobs): $0 2 0.1"
    echo "example (sequential run): $0 1 0.1"
    exit 1
fi

max_jobs=$1
beta=$2

#learning_rates=(0.001 0.0005 0.0001 0.00005 0.00001)
learning_rates=(0.00005 0.00001)

echo "--- starting training runs for beta = ${beta} with a maximum of ${max_jobs} parallel jobs ---"

for lr in "${learning_rates[@]}"; do
    echo "submitting job for lr = ${lr}"

    ./src/vib_mnist_train.py --beta "${beta}" --lr "${lr}" &

    while (( $(jobs -r | wc -l) >= max_jobs )); do
        wait -n
    done
done

echo "--- waiting for remaining jobs to complete... ---"
wait

echo "--- all jobs finished. ---"
