"""
AS-Mamba Testing Script

学習済みAS-Mambaモデルの評価を行うスクリプト
"""

import argparse
import pprint
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.as_mamba_default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_as_mamba import PL_AS_Mamba
from src.utils.profiler import build_profiler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    
    # Model checkpoint
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='path to trained checkpoint')
    
    # Output
    parser.add_argument('--dump_dir', type=str, default='dump/as_mamba_test',
                       help='directory to dump results')
    
    # Performance
    parser.add_argument('--profiler_name', type=str, default='inference',
                       help='profiler: [inference, pytorch] or None')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='batch size (usually 1 for testing)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='number of data loading workers')
    
    # Matching threshold tuning
    parser.add_argument('--thr', type=float, default=None,
                       help='override coarse matching threshold')
    
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    """Main testing function"""
    # Parse arguments
    args = parse_args()
    pprint.pprint(vars(args))
    
    # Load configuration
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    
    # Set seed for reproducibility
    pl.seed_everything(config.TRAINER.SEED)
    
    # Override threshold if specified
    if args.thr is not None:
        config.AS_MAMBA.MATCH_COARSE.THR = args.thr
        loguru_logger.info(f"Overriding matching threshold: {args.thr}")
    
    loguru_logger.info("Configuration loaded")
    
    # Initialize profiler
    profiler = build_profiler(args.profiler_name)
    
    # Initialize model
    model = PL_AS_Mamba(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler,
        dump_dir=args.dump_dir
    )
    loguru_logger.info("AS-Mamba model loaded")
    
    # Initialize data module
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("DataModule initialized")
    
    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        replace_sampler_ddp=False,
        logger=False
    )
    
    # Create dump directory
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        loguru_logger.info(f"Results will be saved to: {args.dump_dir}")
    
    # Run testing
    loguru_logger.info("="*50)
    loguru_logger.info("Starting AS-Mamba testing...")
    loguru_logger.info("="*50)
    
    trainer.test(model, datamodule=data_module, verbose=False)
    
    loguru_logger.info("Testing completed!")
    
    # Print summary
    if args.dump_dir:
        loguru_logger.info(f"\nResults saved to: {args.dump_dir}")
        loguru_logger.info("Files created:")
        loguru_logger.info("  - AS_MAMBA_pred_eval.npy: Prediction results")
        loguru_logger.info("  - AS_MAMBA_flow_stats.npy: Flow statistics")


if __name__ == '__main__':
    main()