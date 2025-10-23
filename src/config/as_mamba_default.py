"""
Default configuration for AS-Mamba model.
Extended from JamMa configuration with AS-Mamba specific parameters.
"""

from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  AS-MAMBA Pipeline  ↓  ##############
_CN.AS_MAMBA = CN()

# Resolution settings (inherited from JamMa)
_CN.AS_MAMBA.RESOLUTION = (8, 2)  # (coarse_scale, fine_scale)
_CN.AS_MAMBA.FINE_WINDOW_SIZE = 5  # Window size for fine-level matching

# Model architecture
_CN.AS_MAMBA.COARSE = CN()
_CN.AS_MAMBA.COARSE.D_MODEL = 96  # Dimension of coarse features

_CN.AS_MAMBA.FINE = CN()
_CN.AS_MAMBA.FINE.D_MODEL = 64  # Dimension of fine features
_CN.AS_MAMBA.FINE.TOP_K = 2
_CN.AS_MAMBA.FINE.ACCEPT_SCORE = 0.2
_CN.AS_MAMBA.FINE.ACCEPT_PEAKINESS = 0.95
_CN.AS_MAMBA.FINE.DSMAX_TEMPERATURE = 0.1 

# AS-Mamba specific configurations
_CN.AS_MAMBA.N_BLOCKS = 3  # Number of AS-Mamba blocks
_CN.AS_MAMBA.D_GEOM = 64  # Dimension of geometric features
_CN.AS_MAMBA.USE_KAN_FLOW = True  # Use KAN for flow prediction (False = MLP)
_CN.AS_MAMBA.GLOBAL_DEPTH = 4  # Number of Mamba layers in global path
_CN.AS_MAMBA.LOCAL_DEPTH = 4  # Number of Mamba layers in local path
_CN.AS_MAMBA.USE_GEOM_FOR_FINE = True  # Use geometric features in fine matching

# Flow predictor settings
_CN.AS_MAMBA.FLOW = CN()
_CN.AS_MAMBA.FLOW.HIDDEN_DIM = 128  # Hidden dimension for flow predictor
_CN.AS_MAMBA.FLOW.NUM_LAYERS = 3  # Number of layers in flow predictor
_CN.AS_MAMBA.FLOW.DROPOUT = 0.1  # Dropout rate

# Adaptive span settings
_CN.AS_MAMBA.ADAPTIVE_SPAN = CN()
_CN.AS_MAMBA.ADAPTIVE_SPAN.BASE_SPAN = 7  # Minimum span size
_CN.AS_MAMBA.ADAPTIVE_SPAN.MAX_SPAN = 15  # Maximum span size
_CN.AS_MAMBA.ADAPTIVE_SPAN.TEMPERATURE = 1.0  # Temperature for span computation

# Matching settings (inherited and extended from JamMa)
_CN.AS_MAMBA.MATCH_COARSE = CN()
_CN.AS_MAMBA.MATCH_COARSE.USE_SM = True  # Use softmax in matching
_CN.AS_MAMBA.MATCH_COARSE.THR = 0.2  # Matching threshold
_CN.AS_MAMBA.MATCH_COARSE.BORDER_RM = 2  # Border removal
_CN.AS_MAMBA.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.AS_MAMBA.MATCH_COARSE.SKH_ITERS = 3
_CN.AS_MAMBA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
_CN.AS_MAMBA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200
_CN.AS_MAMBA.MATCH_COARSE.INFERENCE = False

_CN.AS_MAMBA.FINE = CN()
_CN.AS_MAMBA.FINE.D_MODEL = 64
_CN.AS_MAMBA.FINE.DENSER = False
_CN.AS_MAMBA.FINE.INFERENCE = False
_CN.AS_MAMBA.FINE.DSMAX_TEMPERATURE = 0.1
_CN.AS_MAMBA.FINE.THR = 0.1

# Loss weights
_CN.AS_MAMBA.LOSS = CN()
_CN.AS_MAMBA.LOSS.COARSE_WEIGHT = 1.0
_CN.AS_MAMBA.LOSS.FINE_WEIGHT = 1.0
_CN.AS_MAMBA.LOSS.FLOW_WEIGHT = 0.5  # Weight for flow prediction loss
_CN.AS_MAMBA.LOSS.GEOM_WEIGHT = 0.1  # Weight for geometric consistency loss
_CN.AS_MAMBA.LOSS.FOCAL_ALPHA = 0.25
_CN.AS_MAMBA.LOSS.FOCAL_GAMMA = 2.0
_CN.AS_MAMBA.LOSS.POS_WEIGHT = 1.0
_CN.AS_MAMBA.LOSS.NEG_WEIGHT = 1.0
_CN.AS_MAMBA.LOSS.SUB_WEIGHT = 1e4
_CN.AS_MAMBA.LOSS.EPIPOLAR_WEIGHT = 1.0
_CN.AS_MAMBA.LOSS.MULTISCALE_WEIGHT = 1.0

# Performance settings
_CN.AS_MAMBA.MP = False  # Mixed precision training
_CN.AS_MAMBA.EVAL_TIMES = 1  # Number of evaluation runs

## Memory optimizations
_CN.AS_MAMBA.USE_CHECKPOINT = True
_CN.AS_MAMBA.CHECKPOINT_SEGMENTS = ['flow', 'global', 'local']

##############  Dataset (same as JamMa)  ##############

##############  Dataset  ##############
_CN.DATASET = CN()

# Data source
_CN.DATASET.TRAINVAL_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_SOURCE = None

# Training data paths
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None

# Validation data paths
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_INTRINSIC_PATH = None

# Test data paths
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None
_CN.DATASET.TEST_INTRINSIC_PATH = None

# Dataset configuration
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None
_CN.DATASET.CORR_TH = 5

# MegaDepth specific
_CN.DATASET.MGDPT_IMG_RESIZE = 832
_CN.DATASET.MGDPT_IMG_PAD = True
_CN.DATASET.MGDPT_DEPTH_PAD = True
_CN.DATASET.MGDPT_DF = 8

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 2
_CN.TRAINER.CANONICAL_LR = 1e-4
_CN.TRAINER.SCALING = None
_CN.TRAINER.FIND_LR = False

# Optimizer
_CN.TRAINER.OPTIMIZER = "adamw"
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.
_CN.TRAINER.ADAMW_DECAY = 0.1

# Warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# Learning rate scheduler
_CN.TRAINER.SCHEDULER = 'CosineAnnealing'
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30
_CN.TRAINER.ELR_GAMMA = 0.999992

# Training settings
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.SEED = 66

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 1e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth 

# Evaluation
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32
_CN.TRAINER.PLOT_MODE = 'evaluation'
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
_CN.TRAINER.EPI_ERR_THR = 1e-4
_CN.TRAINER.POSE_GEO_MODEL = 'E'
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# Data sampler
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True
_CN.TRAINER.SB_SUBSET_SHUFFLE = True
_CN.TRAINER.SB_REPEAT = 1


def get_cfg_defaults():
    """Get default config for AS-Mamba."""
    return _CN.clone()


def update_config(config, args):
    """Update config with command line arguments."""
    # This function can be extended to override config with CLI args
    return config