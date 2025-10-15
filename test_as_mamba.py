"""
Test script for AS-Mamba.

Usage:
    # Test on ScanNet
    python test_as_mamba.py \
        configs/data/scannet_test_1500.py \
        configs/as_mamba/indoor/test.py \
        --ckpt_path weights/as_mamba_indoor.ckpt \
        --dump_dir results/as_mamba_scannet \
        --gpus 1

    # Test on MegaDepth
    python test_as_mamba.py \
        configs/data/megadepth_test_1500.py \
        configs/as_mamba/outdoor/test.py \
        --ckpt_path weights/as_mamba_outdoor.ckpt \
        --dump_dir results/as_mamba_megadepth \
        --gpus 1

    # Quick test (no result saving)
    python test_as_mamba.py \
        configs/data/scannet_test_1500.py \
        configs/as_mamba/indoor/test.py \
        --ckpt_path weights/as_mamba_indoor.ckpt \
        --gpus 1
"""

import argparse
import pprint
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.as_mamba_default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_as_mamba import PL_ASMamba
from src.utils.profiler import build_profiler


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='AS-Mamba Test Script'
    )
    
    # Configuration
    parser.add_argument(
        'data_cfg_path', type=str,
        help='Data config path (e.g., configs/data/scannet_test_1500.py)'
    )
    parser.add_argument(
        'main_cfg_path', type=str,
        help='Main config path (e.g., configs/as_mamba/indoor/test.py)'
    )
    
    # Model checkpoint
    parser.add_argument(
        '--ckpt_path', type=str, required=True,
        help='Path to trained AS-Mamba checkpoint'
    )
    
    # Output
    parser.add_argument(
        '--dump_dir', type=str, default=None,
        help='Directory to save matching results and visualizations'
    )
    parser.add_argument(
        '--save_viz', action='store_true',
        help='Save visualization images'
    )
    
    # Runtime options
    parser.add_argument(
        '--profiler_name', type=str, default='inference',
        help='Profiler: [inference, pytorch] or None'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size (typically 1 for testing)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help='Number of data loading workers'
    )
    
    # Matching threshold tuning
    parser.add_argument(
        '--coarse_thr', type=float, default=None,
        help='Override coarse matching threshold'
    )
    parser.add_argument(
        '--fine_thr', type=float, default=None,
        help='Override fine matching threshold'
    )
    
    # Add PyTorch Lightning arguments
    parser = pl.Trainer.add_argparse_args(parser)
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    pprint.pprint(vars(args))
    
    # Load configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    
    # Set seed for reproducibility
    pl.seed_everything(config.TRAINER.SEED)
    
    # Override thresholds if specified
    if args.coarse_thr is not None:
        config.AS_MAMBA.MATCH_COARSE.THR = args.coarse_thr
        loguru_logger.info(f"Overriding coarse threshold: {args.coarse_thr}")
    
    if args.fine_thr is not None:
        config.AS_MAMBA.FINE.THR = args.fine_thr
        loguru_logger.info(f"Overriding fine threshold: {args.fine_thr}")
    
    # Setup output directory
    if args.dump_dir:
        dump_path = Path(args.dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        loguru_logger.info(f"Results will be saved to: {dump_path}")
        
        # Save configuration
        config_save_path = dump_path / 'test_config.yaml'
        with open(config_save_path, 'w') as f:
            f.write(config.dump())
        loguru_logger.info(f"Configuration saved to: {config_save_path}")
    
    # Enable visualization saving
    if args.save_viz:
        config.TRAINER.SAVE_TEST_VIZ = True
    
    loguru_logger.info("Test Configuration:")
    loguru_logger.info(f"  - Checkpoint: {args.ckpt_path}")
    loguru_logger.info(f"  - Dataset: {config.DATASET.TEST_DATA_SOURCE}")
    loguru_logger.info(f"  - Coarse Threshold: {config.AS_MAMBA.MATCH_COARSE.THR}")
    loguru_logger.info(f"  - Fine Threshold: {config.AS_MAMBA.FINE.THR}")
    loguru_logger.info(f"  - AS-Mamba Blocks: {config.AS_MAMBA.N_BLOCKS}")# Initialize profiler
    profiler = build_profiler(args.profiler_name)# Initialize AS-Mamba model
    model = PL_ASMamba(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir
    )
    loguru_logger.info("AS-Mamba LightningModule initialized!")# Initialize data module
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("DataModule initialized!")# Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        replace_sampler_ddp=False,
        logger=False
    )
    loguru_logger.info("=" * 80)
    loguru_logger.info("Starting AS-Mamba Testing!")
    loguru_logger.info("=" * 80)# Run test
    trainer.test(model, datamodule=data_module, verbose=False)
    loguru_logger.info("=" * 80)
    loguru_logger.info("Testing completed!")
    loguru_logger.info("=" * 80)
    if __name__ == 'main':
        main()

