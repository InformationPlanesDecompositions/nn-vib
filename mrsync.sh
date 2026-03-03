#!/bin/sh

dest="conqueror-home:dev/nn-vib"

if [ "$1" = "--away" ]; then
    dest="ln@conqueror.lneural.net:dev/nn-vib"
elif [ -n "$1" ]; then
    echo "usage: $0 [--away]"
    exit 1
fi

rsync -av \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='*.pth' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='vib_lenet_old/' \
    --exclude='save_stats_weights' \
    --exclude='plots/' \
    --exclude='saved_plots/' \
    ./ "$dest"
