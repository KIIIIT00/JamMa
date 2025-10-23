#!/bin/bash
# Quick debug script for AS-Mamba

echo "=== AS-Mamba Quick Debug ==="
echo "This will run a fast training iteration for debugging"

python train_as_mamba.py \
    configs/data/megadepth_trainval_832.py \
    configs/as_mamba/outdoor/debug.py \
    --exp_name debug_$(date +%Y%m%d_%H%M%S) \
    --gpus 1 \
    --batch_size 1 \
    --num_workers 2 \
    --max_epochs 2 \
    --limit_train_batches 10 \
    --limit_val_batches 10 \
    --val_check_interval 5 \
    --precision 16 \
    --debug

echo "Debug run completed!"