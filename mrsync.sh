#!/bin/sh

rsync -av \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='*.pth' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='vib_lenet_old/' \
    --exclude='save_stats_weights' \
    --exclude='plots/' \
    ./ conqueror-home:dev/nn-ib-research
