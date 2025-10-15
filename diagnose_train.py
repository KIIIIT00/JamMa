# diagnose_train.py
import argparse
import sys
from src.config.as_mamba_default import get_cfg_defaults
from src.utils.misc import setup_gpus

# Simulate the training script
parser = argparse.ArgumentParser()
parser.add_argument('data_cfg_path', type=str, default='configs/data/megadepth_trainval_832.py')
parser.add_argument('main_cfg_path', type=str, default='configs/as_mamba/outdoor/train.py')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gpus', type=int, default=4)
parser = argparse.ArgumentParser().parse_args.__self__.add_argument
# Use the actual parser from train script
import train_as_mamba
args_string = [
    'configs/data/megadepth_trainval_832.py',
    'configs/as_mamba/outdoor/train.py',
    '--gpus', '4',
    '--batch_size', '2'
]

# Mock parse
class MockArgs:
    data_cfg_path = 'configs/data/megadepth_trainval_832.py'
    main_cfg_path = 'configs/as_mamba/outdoor/train.py'
    batch_size = 2
    gpus = 4
    num_nodes = None  # This might be the issue

args = MockArgs()

print("=" * 80)
print("DIAGNOSIS: Training Configuration")
print("=" * 80)

# Load config
config = get_cfg_defaults()
print(f"✓ Default config loaded")
print(f"  CANONICAL_BS (default): {config.TRAINER.CANONICAL_BS}")

config.merge_from_file(args.main_cfg_path)
print(f"✓ Main config merged: {args.main_cfg_path}")
print(f"  CANONICAL_BS (after main): {config.TRAINER.CANONICAL_BS}")

config.merge_from_file(args.data_cfg_path)
print(f"✓ Data config merged: {args.data_cfg_path}")
print(f"  CANONICAL_BS (after data): {config.TRAINER.CANONICAL_BS}")

# Setup GPUs
print("\n" + "=" * 80)
print("GPU Setup:")
print("=" * 80)
_n_gpus = setup_gpus(args.gpus)
print(f"  args.gpus: {args.gpus}")
print(f"  _n_gpus (actual): {_n_gpus}")
print(f"  args.num_nodes: {args.num_nodes}")

# Check num_nodes
if not hasattr(args, 'num_nodes') or args.num_nodes is None:
    print(f"  ⚠️  num_nodes is None or not set!")
    args.num_nodes = 1
    print(f"  Setting num_nodes to default: {args.num_nodes}")

# Calculate
print("\n" + "=" * 80)
print("Batch Size Calculation:")
print("=" * 80)
WORLD_SIZE = _n_gpus * args.num_nodes
TRUE_BATCH_SIZE = WORLD_SIZE * args.batch_size

print(f"  WORLD_SIZE = {_n_gpus} GPUs × {args.num_nodes} nodes = {WORLD_SIZE}")
print(f"  TRUE_BATCH_SIZE = {WORLD_SIZE} × {args.batch_size} = {TRUE_BATCH_SIZE}")
print(f"  CANONICAL_BS = {config.TRAINER.CANONICAL_BS}")

# Check for zero
print("\n" + "=" * 80)
print("Validation:")
print("=" * 80)
if TRUE_BATCH_SIZE == 0:
    print("❌ ERROR: TRUE_BATCH_SIZE is 0!")
    print("   This will cause division by zero.")
    sys.exit(1)
elif config.TRAINER.CANONICAL_BS == 0:
    print("❌ ERROR: CANONICAL_BS is 0!")
    print("   This will cause division by zero.")
    sys.exit(1)
else:
    print("✓ Both values are valid")

# Calculate scaling
_scaling = TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
print(f"\n  _scaling = {TRUE_BATCH_SIZE} / {config.TRAINER.CANONICAL_BS} = {_scaling}")

if _scaling == 0:
    print("❌ ERROR: Scaling factor is 0!")
    print("   This should not happen if TRUE_BATCH_SIZE and CANONICAL_BS are both > 0")
    sys.exit(1)
else:
    print(f"✓ Scaling factor is valid: {_scaling}")

print("\n" + "=" * 80)
print("✓ All checks passed!")
print("=" * 80)