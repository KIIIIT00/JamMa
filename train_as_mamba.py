"""
Training script for AS-Mamba.

Usage:
    # Debug mode (quick iteration)
    python train_as_mamba.py \
        configs/data/scannet_trainval.py \
        configs/as_mamba/indoor/debug.py \
        --exp_name as_mamba_debug \
        --gpus 1 \
        --batch_size 1 \
        --num_workers 2 \
        --max_epochs 5

    # Full training (indoor)
    python train_as_mamba.py \
        configs/data/scannet_trainval.py \
        configs/as_mamba/indoor/train.py \
        --exp_name as_mamba_indoor \
        --gpus 4 \
        --batch_size 2 \
        --num_workers 4 \
        --max_epochs 30

    # Full training (outdoor)
    python train_as_mamba.py \
        configs/data/megadepth_trainval_832.py \
        configs/as_mamba/outdoor/train.py \
        --exp_name as_mamba_outdoor \
        --gpus 4 \
        --batch_size 2 \
        --num_workers 4 \
        --max_epochs 30
"""
import torch
import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.as_mamba_default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_as_mamba import PL_ASMamba


loguru_logger = get_rank_zero_only_logger(loguru_logger)

def train_step_with_memory_logging(self, batch, batch_idx):
    if batch_idx % 10 == 0:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        memory_stats = torch.cuda.memory_stats()
        active_bytes = memory_stats["active_bytes.all.current"] / 1024**3
        
        loguru_logger.info(f"GPU Memory Stats [Step {self.global_step}]: "
                   f"Allocated: {allocated:.2f} GB, "
                   f"Reserved: {reserved:.2f} GB, "
                   f"Max Allocated: {max_allocated:.2f} GB, "
                   f"Active: {active_bytes:.2f} GB")
        
    return super().training_step(batch, batch_idx)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='AS-Mamba Training Script'
    )
    
    # Configuration files
    parser.add_argument(
        'data_cfg_path', type=str, 
        help='Data config path (e.g., configs/data/scannet_trainval.py)'
    )
    parser.add_argument(
        'main_cfg_path', type=str,
        help='Main config path (e.g., configs/as_mamba/indoor/train.py)'
    )
    
    # Experiment settings
    parser.add_argument(
        '--exp_name', type=str, default='as_mamba_exp',
        help='Experiment name for logging'
    )
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True,
        help='Pin memory for data loading'
    )
    
    # Model loading
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='Checkpoint path for resuming or fine-tuning'
    )
    parser.add_argument(
        '--pretrained_jamma', type=str, default=None,
        help='Path to pretrained JamMa weights for initialization'
    )
    
    # Training options
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='Disable checkpoint saving (for debugging)'
    )
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='Profiler: [inference, pytorch] or None'
    )
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='Load datasets in parallel'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode (fast_dev_run)'
    )
    
    # Add PyTorch Lightning trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))
    
    # Load configuration
    config = get_cfg_defaults()
    loguru_logger.info(f"[ CONFIG ] cfg: {config}")
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    
    # Set random seed for reproducibility
    pl.seed_everything(config.TRAINER.SEED)
    
    if not hasattr(args, 'num_nodes') or args.num_nodes is None:
        args.num_nodes = 1
        loguru_logger.info("num_nodes not specified, defaulting to: 1")
    
    # Setup distributed training
    # args.gpus = _n_gpus = setup_gpus(args.gpus)
    # config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    # config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    
    # # Scale learning rate and warmup steps
    # _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    # args.gpus = _n_gpus = setup_gpus(args.gpus)
    # config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    # config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    
    # # WORKAROUND: Ensure CANONICAL_BS has a valid default
    # if not hasattr(config.TRAINER, 'CANONICAL_BS') or config.TRAINER.CANONICAL_BS <= 0:
    #     loguru_logger.warning("⚠️  CANONICAL_BS not set or invalid, using default: 2")
    #     config.TRAINER.CANONICAL_BS = 2
    
    # # Scale learning rate and warmup steps
    # _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    # config.TRAINER.SCALING = _scaling
    # config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    # config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # Setup distributed training
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    
    # Ensure num_nodes has a valid default
    if not hasattr(args, 'num_nodes') or args.num_nodes is None:
        args.num_nodes = 1
        loguru_logger.info("num_nodes not specified, using default: 1")
    
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    
    # Debug logging
    loguru_logger.info("=" * 80)
    loguru_logger.info("Batch Size Configuration:")
    loguru_logger.info(f"  - GPUs: {_n_gpus}")
    loguru_logger.info(f"  - Nodes: {args.num_nodes}")
    loguru_logger.info(f"  - Batch size per GPU: {args.batch_size}")
    loguru_logger.info(f"  - WORLD_SIZE: {config.TRAINER.WORLD_SIZE}")
    loguru_logger.info(f"  - TRUE_BATCH_SIZE: {config.TRAINER.TRUE_BATCH_SIZE}")
    loguru_logger.info(f"  - CANONICAL_BS: {config.TRAINER.CANONICAL_BS}")
    
    # Validation
    if config.TRAINER.TRUE_BATCH_SIZE <= 0:
        loguru_logger.error("❌ TRUE_BATCH_SIZE must be > 0")
        loguru_logger.error(f"   Current value: {config.TRAINER.TRUE_BATCH_SIZE}")
        loguru_logger.error(f"   = WORLD_SIZE({config.TRAINER.WORLD_SIZE}) × batch_size({args.batch_size})")
        raise ValueError("Invalid TRUE_BATCH_SIZE")
    
    if config.TRAINER.CANONICAL_BS <= 0:
        loguru_logger.error("❌ CANONICAL_BS must be > 0")
        loguru_logger.error(f"   Current value: {config.TRAINER.CANONICAL_BS}")
        raise ValueError("Invalid CANONICAL_BS")
    
    # Scale learning rate and warmup steps
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    loguru_logger.info(f"  - Scaling Factor: {_scaling:.4f}")
    loguru_logger.info(f"  - TRUE_LR: {config.TRAINER.TRUE_LR:.2e}")
    loguru_logger.info(f"  - WARMUP_STEP: {config.TRAINER.WARMUP_STEP}")
    loguru_logger.info("=" * 80)
    
    loguru_logger.info(f"Training Configuration:")
    loguru_logger.info(f"  - World Size: {config.TRAINER.WORLD_SIZE}")
    loguru_logger.info(f"  - Batch Size (total): {config.TRAINER.TRUE_BATCH_SIZE}")
    loguru_logger.info(f"  - Learning Rate: {config.TRAINER.TRUE_LR:.2e}")
    loguru_logger.info(f"  - Warmup Steps: {config.TRAINER.WARMUP_STEP}")
    loguru_logger.info(f"  - AS-Mamba Blocks: {config.AS_MAMBA.N_BLOCKS}")
    loguru_logger.info(f"  - Flow Weight: {config.AS_MAMBA.LOSS.FLOW_WEIGHT}")
    
    # Initialize profiler
    profiler = build_profiler(args.profiler_name)
    
    # Initialize AS-Mamba Lightning module
    model = PL_ASMamba(
        config,
        pretrained_ckpt=args.ckpt_path,
        profiler=profiler
    )
    loguru_logger.info("AS-Mamba LightningModule initialized!")
    
    # Initialize data module
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("DataModule initialized!")
    
    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir='as_mamba_logs/',
        name=args.exp_name,
        default_hp_metric=False
    )
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Model checkpoint
    if not args.disable_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='auc@10',
            verbose=True,
            save_top_k=3,
            mode='max',
            save_last=True,
            dirpath=str(ckpt_dir),
            filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}'
        )
        callbacks.append(ckpt_callback)
        loguru_logger.info(f"Checkpoints will be saved to: {ckpt_dir}")
    
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(
            find_unused_parameters=True,
            num_nodes=args.num_nodes,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0
        ),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,
        reload_dataloaders_every_epoch=False,
        weights_summary='full',
        profiler=profiler,
        fast_dev_run=args.debug,
        precision=16,
        accumulate_grad_batches=2
    )
    
    loguru_logger.info("=" * 80)
    loguru_logger.info("Starting AS-Mamba Training!")
    loguru_logger.info("=" * 80)
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    loguru_logger.info("Training completed!")


if __name__ == '__main__':
    main()