"""
Debug configuration for AS-Mamba (Indoor/ScanNet)
Fast training with reduced data for quick iteration.
"""

from configs.as_mamba.base import cfg

# Reduce model complexity for faster training
cfg.AS_MAMBA.N_BLOCKS = 2  # Reduced from 3
cfg.AS_MAMBA.GLOBAL_DEPTH = 2  # Reduced from 4
cfg.AS_MAMBA.LOCAL_DEPTH = 2   # Reduced from 4

# Training settings for quick debugging
cfg.TRAINER.CANONICAL_BS = 1
cfg.TRAINER.CANONICAL_LR = 5e-5
cfg.TRAINER.WARMUP_STEP = 100
cfg.TRAINER.COSA_TMAX = 5  # Short training
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 50  # Very small dataset
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 4

# Pose estimation (indoor)
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.EPI_ERR_THR = 5e-4  # ScanNet threshold

# Matching
cfg.AS_MAMBA.MATCH_COARSE.INFERENCE = False
cfg.AS_MAMBA.FINE.INFERENCE = False
cfg.AS_MAMBA.MATCH_COARSE.THR = 0.2
cfg.AS_MAMBA.FINE.THR = 0.1
cfg.AS_MAMBA.MATCH_COARSE.BORDER_RM = 2