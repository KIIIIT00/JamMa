"""
Debug configuration for AS-Mamba (Outdoor/MegaDepth)
"""

from configs.as_mamba.base import cfg

# Reduced model for debugging
cfg.AS_MAMBA.N_BLOCKS = 2
cfg.AS_MAMBA.GLOBAL_DEPTH = 2
cfg.AS_MAMBA.LOCAL_DEPTH = 2
cfg.AS_MAMBA.MP = True

## Memory optimizations
cfg.AS_MAMBA.USE_CHECKPOINT = True
cfg.AS_MAMBA.CHECKPOINT_SEGMENTS = ['flow', 'global', 'local']

# Training settings
cfg.TRAINER.CANONICAL_BS = 1
cfg.TRAINER.CANONICAL_LR = 5e-5
cfg.TRAINER.WARMUP_STEP = 100
cfg.TRAINER.COSA_TMAX = 5
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 50
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 4

# Outdoor-specific
cfg.TRAINER.EPI_ERR_THR = 1e-4  # MegaDepth threshold
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.AS_MAMBA.MATCH_COARSE.THR = 0.2
cfg.AS_MAMBA.FINE.THR = 0.1