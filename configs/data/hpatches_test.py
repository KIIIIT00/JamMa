from configs.data.base import cfg

cfg.DATASET.TEST_DATA_SOURCE = "HPatches"
cfg.DATASET.TEST_DATA_ROOT = "data/hpatches-sequences-release"

# HPatches specific settings
cfg.DATASET.HPATCHES_ALTERATION = 'all'  # 'all', 'v', or 'i'
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0  # Not used for HPatches

# Use similar settings to MegaDepth for consistency
cfg.DATASET.MGDPT_IMG_RESIZE = 480  # Smaller size for HPatches
cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 8