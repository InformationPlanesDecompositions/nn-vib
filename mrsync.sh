#!/bin/sh

dest="conqueror:dev/nn-vib"

if [ "$1" = "--away" ]; then
  dest="ln@conqueror-away:dev/nn-vib"
elif [ -n "$1" ]; then
  echo "usage: $0 [--away]"
  exit 1
fi

#--exclude='*.pth' \
rsync -av \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='vib_lenet_old/' \
  --exclude='save_stats_weights/' \
  --exclude='plots/' \
  --exclude='saved_plots/' \
  --exclude='mrsync.sh' \
  ./ "$dest"
