"""
Test configuration for AS-Mamba (Outdoor/MegaDepth)
"""

from configs.as_mamba.base import cfg

# Test settings
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.EPI_ERR_THR = 1e-4  # MegaDepth threshold

# Inference mode
cfg.AS_MAMBA.MP = False
cfg.AS_MAMBA.EVAL_TIMES = 1
cfg.AS_MAMBA.MATCH_COARSE.INFERENCE = True
cfg.AS_MAMBA.FINE.INFERENCE = True
cfg.AS_MAMBA.MATCH_COARSE.USE_SM = True
cfg.AS_MAMBA.MATCH_COARSE.THR = 0.2
cfg.AS_MAMBA.FINE.THR = 0.1
cfg.AS_MAMBA.MATCH_COARSE.BORDER_RM = 2
