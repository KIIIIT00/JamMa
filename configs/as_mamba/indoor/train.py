"""
Full training configuration for AS-Mamba (Indoor/ScanNet)
"""

from configs.as_mamba.base import cfg

# Full model configuration
cfg.AS_MAMBA.N_BLOCKS = 3
cfg.AS_MAMBA.GLOBAL_DEPTH = 4
cfg.AS_MAMBA.LOCAL_DEPTH = 4
cfg.AS_MAMBA.D_GEOM = 64

# Training settings
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.WARMUP_STEP = 4800  # ~3 epochs on ScanNet
cfg.TRAINER.SCHEDULER = 'CosineAnnealing'
cfg.TRAINER.COSA_TMAX = 30
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 200

# Indoor-specific settings
cfg.TRAINER.EPI_ERR_THR = 5e-4  # ScanNet threshold
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

# Matching configuration
cfg.AS_MAMBA.MATCH_COARSE.THR = 0.2
cfg.AS_MAMBA.FINE.THR = 0.1
cfg.AS_MAMBA.MATCH_COARSE.BORDER_RM = 2