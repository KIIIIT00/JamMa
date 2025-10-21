"""
Minimal training configuration for debugging AS-Mamba
- 最小データセット
- 少ないエポック数
- メモリプロファイリング有効
"""

_base_ = ['../base.py']

# AS-Mamba specific debug settings
AS_MAMBA = dict(
    N_BLOCKS = 1,  # 最小ブロック数でメモリ使用量削減
    RESOLUTION = (8, 2),
    FINE_WINDOW_SIZE = 3,  # ウィンドウサイズを小さく
    
    # メモリ最適化
    USE_CHECKPOINT = True,  # Gradient checkpointing有効
    MP = False,  # Mixed precisionは一旦無効（デバッグ用）
    
    # 適応的スパン設定を最小化
    ADAPTIVE_SPAN = dict(
        BASE_SPAN = 3,  # 小さいベーススパン
        MAX_SPAN = 7,   # 最大スパンも制限
        TEMPERATURE = 1.0
    ),
    
    # 損失重みの調整（デバッグ用）
    LOSS = dict(
        FLOW_WEIGHT = 0.1,  # フロー損失を軽く
        COARSE_WEIGHT = 1.0,
        FINE_WEIGHT = 0.5,
        EPIPOLAR_WEIGHT = 0.0,  # 一時的に無効
        MULTISCALE_WEIGHT = 0.0  # 一時的に無効
    )
)

# トレーニング設定
TRAINER = dict(
    SEED = 42,
    WORLD_SIZE = 1,
    CANONICAL_BS = 1,
    CANONICAL_LR = 1e-4,
    WARMUP_STEP = 10,
    WARMUP_RATIO = 0.1,
    SCHEDULER = 'CosineAnnealing',
    COSA_TMAX = 5,  # 短いサイクル
    
    # デバッグ用設定
    N_VAL_PAIRS_TO_PLOT = 2,
    GRADIENT_CLIPPING = 1.0,
    
    # メトリクス
    EPI_ERR_THR = 1e-3,  # 緩い閾値
    RANSAC_PIXEL_THR = 1.0,
    POSE_ESTIMATION_METHOD = 'RANSAC'  # シンプルな手法
)

# データセット設定（最小）
DATASET = dict(
    MIN_OVERLAP_SCORE_TRAIN = 0.6,  # より厳しい条件で少ないデータ
    MIN_OVERLAP_SCORE_TEST = 0.5,
    AUGMENTATION_TYPE = None,  # データ拡張無効
    MGDPT_IMG_RESIZE = 480,  # 小さい画像サイズ
    MGDPT_IMG_PAD = True,
    MGDPT_DEPTH_PAD = True,
    MGDPT_DF = 16  # より粗い解像度
)