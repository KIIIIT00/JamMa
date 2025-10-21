from configs.as_mamba.base import cfg

# モデル複雑度の大幅削減
cfg.AS_MAMBA.N_BLOCKS = 1  # 1ブロックのみ
cfg.AS_MAMBA.GLOBAL_DEPTH = 2  # グローバルパス深さ削減
cfg.AS_MAMBA.LOCAL_DEPTH = 2   # ローカルパス深さ削減
cfg.AS_MAMBA.D_GEOM = 32  # 特徴次元削減

# コアモデル次元の削減
cfg.AS_MAMBA.COARSE.D_MODEL = 128  # 粗レベル特徴次元削減
cfg.AS_MAMBA.FINE.D_MODEL = 32  # 細レベル特徴次元削減

# チェックポイントによるメモリ最適化を有効化
cfg.AS_MAMBA.USE_CHECKPOINT = True

# メモリ最適化のための学習設定
cfg.TRAINER.CANONICAL_BS = 1
cfg.TRAINER.CANONICAL_LR = 1e-5
cfg.TRAINER.WARMUP_STEP = 50
cfg.TRAINER.COSA_TMAX = 3  # 非常に短い学習期間

# データセット削減
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 20  # 非常に小さいデータセット
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 2