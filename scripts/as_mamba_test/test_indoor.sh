#!/bin/bash
# Test script for AS-Mamba (Indoor/ScanNet)

echo "=== AS-Mamba Indoor Testing ==="

# Configuration
CKPT_PATH=${1:-"weights/as_mamba_indoor_best.ckpt"}
DUMP_DIR="results/as_mamba_scannet_$(date +%Y%m%d_%H%M%S)"

echo "Checkpoint: $CKPT_PATH"
echo "Output Directory: $DUMP_DIR"

python test_as_mamba.py \
    configs/data/scannet_test_1500.py \
    configs/as_mamba/indoor/test.py \
    --ckpt_path $CKPT_PATH \
    --dump_dir $DUMP_DIR \
    --save_viz \
    --gpus 1 \
    --batch_size 1 \
    --num_workers 2

echo "Testing completed!"
echo "Results saved to: $DUMP_DIR"