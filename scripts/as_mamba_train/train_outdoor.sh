#!/bin/bash
# Full training script for AS-Mamba (Outdoor/MegaDepth)

echo "=== AS-Mamba Outdoor Training ==="

# Configuration
EXP_NAME="as_mamba_outdoor_$(date +%Y%m%d_%H%M%S)"
GPUS=1
BATCH_SIZE=1
NUM_WORKERS=4
MAX_EPOCHS=3

echo "Experiment: $EXP_NAME"
echo "GPUs: $GPUS"
echo "Batch Size: $BATCH_SIZE per GPU"
echo "Max Epochs: $MAX_EPOCHS"

python train_as_mamba.py \
    configs/data/megadepth_trainval_832.py \
    configs/as_mamba/outdoor/train.py \
    --exp_name $EXP_NAME \
    --gpus $GPUS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_epochs $MAX_EPOCHS \
    --accelerator ddp

echo "Training completed!"
echo "Logs saved to: as_mamba_logs/$EXP_NAME"
