#!/bin/sh

rsync -av --exclude='.git/' --exclude='.venv/' --exclude='*.pth' --exclude='__pycache__/' --exclude='.ipynb_checkpoints/' ./ conqueror-home:dev/nn-ib-research
