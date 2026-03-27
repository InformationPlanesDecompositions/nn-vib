#!/bin/sh

betas_override=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --betas)
            shift
            if [ -z "$1" ]; then
                echo "error: --betas requires a value"
                exit 1
            fi
            betas_override="$1"
            shift
            ;;
        -h|--help)
            echo "usage: $0 [--betas <list>] <max_parallel_jobs> <seed1> [seed2 ...]"
            echo "  --betas accepts comma or space separated values"
            echo "example: $0 --betas 0.5,0.4,0.3 3 7 11 42"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt 2 ]; then
    echo "usage: $0 [--betas <list>] <max_parallel_jobs> <seed1> [seed2 ...]"
    echo "example: $0 --betas 0.5,0.4,0.3 3 7 11 42"
    exit 1
fi

max_jobs="$1"
shift

case "$max_jobs" in
    ''|*[!0-9]*|0)
        echo "error: <max_parallel_jobs> must be a positive integer"
        exit 1
        ;;
esac

epochs="400"
lr="2e-4"
betas_small="0.5 0.4 0.3 0.2 0.1 0.01 0.001 0.0001"
betas_large="0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01"
if [ -n "$betas_override" ]; then
    betas_override=$(printf "%s" "$betas_override" | tr ',' ' ')
    betas_small="$betas_override"
    betas_large="$betas_override"
fi
active_jobs=0
fail_marker=".mlp_ib_seed_sweep_failed.$$"
rm -f "$fail_marker"

run_training_job() {
    seed="$1"
    hidden1="$2"
    z_dim="$3"
    hidden2="$4"
    beta="$5"
    base_dir="save_stats_weights/vib_mlp_${hidden1}_${hidden2}_${z_dim}_${beta}_${lr}_${epochs}"
    seed_dir="${base_dir}_seed_${seed}"
    backup_dir=""
    if [ -d "$seed_dir" ]; then
        echo "[skip] seed=${seed} h1=${hidden1} z=${z_dim} h2=${hidden2} beta=${beta} (already exists: ${seed_dir})"
        return 0
    fi
    if [ -d "$base_dir" ]; then
        backup_dir="${base_dir}_preseed_backup"
        suffix=1
        while [ -e "$backup_dir" ]; do
            backup_dir="${base_dir}_preseed_backup_${suffix}"
            suffix=$((suffix + 1))
        done
        mv "$base_dir" "$backup_dir"
    fi
    echo "[start] seed=${seed} h1=${hidden1} z=${z_dim} h2=${hidden2} beta=${beta}"
    ./src/mlp_ib.py \
        --lr "$lr" \
        --epochs "$epochs" \
        --beta "$beta" \
        --hidden1 "$hidden1" \
        --z_dim "$z_dim" \
        --hidden2 "$hidden2" \
        --rnd_seed "$seed"
    status=$?
    if [ "$status" -eq 0 ] && [ -d "$base_dir" ]; then
        mv "$base_dir" "$seed_dir"
        echo "[done]  seed=${seed} h1=${hidden1} z=${z_dim} h2=${hidden2} beta=${beta}"
    else
        echo "[fail]  seed=${seed} h1=${hidden1} z=${z_dim} h2=${hidden2} beta=${beta}"
        : > "$fail_marker"
    fi
    if [ -n "$backup_dir" ] && [ -d "$backup_dir" ]; then
        mv "$backup_dir" "$base_dir"
    fi
    return "$status"
}

echo "--- starting mlp_ib seed sweep (epochs=${epochs}, lr=${lr}, max_jobs=${max_jobs}) ---"

for seed in "$@"; do
    echo "--- running seed ${seed} ---"
    for beta in $betas_small; do
        run_training_job "$seed" 386 15 128 "$beta" &
        active_jobs=$((active_jobs + 1))
        if [ "$active_jobs" -ge "$max_jobs" ]; then
            wait
            active_jobs=0
        fi
    done
    for beta in $betas_small; do
        run_training_job "$seed" 256 10 64 "$beta" &
        active_jobs=$((active_jobs + 1))
        if [ "$active_jobs" -ge "$max_jobs" ]; then
            wait
            active_jobs=0
        fi
    done
    for beta in $betas_large; do
        run_training_job "$seed" 512 10 128 "$beta" &
        active_jobs=$((active_jobs + 1))
        if [ "$active_jobs" -ge "$max_jobs" ]; then
            wait
            active_jobs=0
        fi
    done
    wait
    active_jobs=0
done

if [ ! -f "$fail_marker" ]; then
    echo "--- all jobs finished successfully ---"
    rm -f "$fail_marker"
    exit 0
fi

echo "--- finished with at least one failed job ---"
rm -f "$fail_marker"
exit 1
