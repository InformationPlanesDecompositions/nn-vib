#!/bin/sh

rsync -av \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='*.pth' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='vib_lenet_old/' \
  ./ conqueror-home:dev/nn-ib-research
