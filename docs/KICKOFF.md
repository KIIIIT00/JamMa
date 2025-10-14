# AS-Mamba 研究開発キックオフドキュメント

**バージョン**: 1.1 (Tensor Shape情報を更新)
**作成日**: 2025/09/26

---

## 1. プロジェクト概要 (Project Overview)

### 1.1. 仮説 (Hypothesis)

ASpanFormerの持つ、マッチングの難易度に応じて動的に処理範囲を調整する**階層的・適応的アーキテクチャ**と、JamMaで示されたMambaベースの**高効率・高性能な特徴量相互作用モデル**を融合させることで、既存のあらゆる手法を超える**最高精度**と**リアルタイム性**を両立した、次世代の画像マッチングモデルを構築できる。

### 1.2. 最終目的 (Ultimate Objective)

* **精度**: 主要なベンチマーク（MegaDepth, ScanNet等）において、ASpanFormerを上回るState-of-the-Art（SOTA）精度を達成する。
* **効率**: JamMaに匹敵する、あるいはそれ以上の実行速度（レイテンシ）とモデル効率（パラメータ数、FLOPs）を実現し、高解像度画像にも対応可能な実用性を確保する。
* **革新性**: 「マルチヘッドMamba状態」や「ヒルベルト曲線スキャン」といった新規技術の有効性を実証し、分野に新たな知見を提供する。

---

## 2. アーキテクチャ詳細 (Architecture Details)

* **変数定義**:
    * `B`: バッチサイズ
    * `H, W`: 入力画像の高さ、幅
    * `C_enc`: Encoderが出力する特徴チャネル数 (例: 128)
    * `C_match`: マッチングヘッドが出力する特徴チャネル数 (例: 128)
    * `C_geom`: 幾何ヘッドが出力する特徴チャネル数 (例: 32)
    * `C_fine`: Encoderが出力する詳細特徴のチャネル数 (例: 64)

### 2.1. Feature Encoder

* **役割**: CNNを用いて画像から階層的な局所特徴を抽出する。
* **実装**: **JamMa**の`ConvNeXtV2` (`backbone.py`) を参考に、軽量性と性能を両立したエンコーダを採用する。
* **入力**: 画像ペア `(B, 3, H, W)`
* **出力**:
    * **Coarse Features**: `(B, C_enc, H/8, W/8)`
    * **Fine Features**: `(B, C_fine, H/2, W/2)` (最終的な精密化で使用)

### 2.2. Mamba Initializer

* **役割**: 2枚の画像の特徴量間で初期的な大域情報交換を行い、後段の処理に必要なマッチング特徴と幾何特徴を生成する。
* **実装**: **JamMa**の`JointMamba` (`mamba_module.py`) をベースに、**Multi-Head**化する。
* **入力**: `Coarse Features` `(B, C_enc, H/8, W/8)`
* **出力**:
    * **Matching Features `F_match^1`**: `(B, C_match, H/8, W/8)`
    * **Geometry Features `F_geom^1`**: `(B, C_geom, H/8, W/8)`

### 2.3. AS-Mamba Block (i番目)

* **役割**: アーキテクチャの中核。Flow予測、適応的な局所処理、大域的な文脈集約を反復的に行い、特徴量を精密化する。
* **入力**:
    * 前のブロックからの`Matching Features` `F_match^(i-1)`: `(B, C_match, H/8, W/8)`
    * 前のブロックからの`Geometry Features` `F_geom^(i-1)`: `(B, C_geom, H/8, W/8)`
* **内部コンポーネント**:
    1.  **Flow予測 (KAN)**:
        * **入力**: `F_match^(i-1)` と `F_geom^(i-1)` を結合したもの `(B, C_match + C_geom, H/8, W/8)`
        * **出力**: **ASpanFormer**と同様のFlow Map `Φ`: `(B, H/8, W/8, 4)` (x, y, σ_x, σ_y)
    2.  **Global Path (JEGO-Mamba)**:
        * **入力**: ダウンサンプルされた`F_match^(i-1)` `(B, C_match, H_coarse, W_coarse)` (例: H/32, W/32)
        * **出力**: マッチング/幾何特徴
    3.  **Local Mamba Module**:
        * **入力**: `F_match^(i-1)` `(B, C_match, H/8, W/8)` と Flow Map `Φ`
        * **処理**: `Φ`から計算された適応スパン内で、**Multi-Head Mamba**による局所情報交換を行う。
        * **出力**: マッチング/幾何特徴
    4.  **特徴集約 (FFN)**:
        * **入力**: GlobalとLocalからのマッチング/幾何特徴、およびスキップ接続 `F_match^(i-1)`
        * **処理**: **ASpanFormer**のFFN (`transformer.py`内の`SimpleFeadForward`) を参考に、ConvベースのFFNで特徴を統合。
* **出力**:
    * **Matching Features `F_match^i`**: `(B, C_match, H/8, W/8)`
    * **Geometry Features `F_geom^i`**: `(B, C_geom, H/8, W/8)`

### 2.4. Matching Module

* **役割**: 最終的に得られたマッチング特徴から、信頼性の高い対応点を見つけ出し、サブピクセルレベルまで精密化する。
* **実装**: **JamMa** (`matching_module.py`) と **ASpanFormer** (`coarse_matching.py`, `fine_matching.py`) の両方の実装を参考に、より頑健なものを選択または融合する。
* **入力**:
    * 最終ブロックからの`Matching Features` `F_match^N`: `(B, C_match, H/8, W/8)`
    * Encoderからの`Fine Features`: `(B, C_fine, H/2, W/2)`
* **出力**:
    * **Coarse Matches**: 辞書形式 `{'mkpts0_c': (N_c, 2), 'mkpts1_c': (N_c, 2), ...}`
    * **Fine Matches**: 辞書形式 `{'mkpts0': (N_f, 2), 'mkpts1': (N_f, 2), ...}`
    * 最終的な対応点座標は、これらの辞書から抽出される。形状はマッチした点の数 `N_f` に依存する。

---

## 3. 参照論文の要点 (Key Points from Reference Papers)

### 3.1. ASpanFormer

* **階層構造**: Coarse (Global), Medium, Fine (Local) の3レベルでAttentionを適用し、大域的文脈と局所的詳細を両立。
* **適応スパン**: Flow Mapから予測した**不確実性 (Uncertainty)** に基づき、Local Attentionの範囲を動的に変更。確信度が高い領域は狭く、低い領域は広く探索する。
* **反復的精密化**: `特徴 → Flow予測 → 特徴更新` のループを繰り返すことで、対応点の推定精度を徐々に高めていく。

### 3.2. JamMa

* **Mambaの採用**: TransformerのAttentionをState Space Model (Mamba) に置き換えることで、計算量を $O(N^2)$ から $O(N)$ に削減し、劇的な効率化を達成。
* **JEGO戦略**: Joint Scan (高頻度な相互作用)、Efficient Four-directional Scan (全方位性)、Global Omnidirectional features (Aggregatorによる大域化) を組み合わせた高効率・高性能なスキャン・マージ戦略。
* **超軽量設計**: 少ないパラメータ数で、重量級のモデルに匹敵する性能を達成。

---

## 4. 追加項目 (Additional Sections)

### 4.1. 損失関数 (Loss Function)

* **$L_{total} = w_1 * L_{coarse} + w_2 * L_{fine} + w_3 * L_{flow} + w_4 * L_{geom}$**
    * `L_coarse`, `L_fine`: 粗い/詳細なマッチングの正解との誤差（**JamMa**の`FocalLoss` (`loss.py`) を参考）。
    * `L_flow`: KANが予測したFlow Mapの正解との誤差（**ASpanFormer**の`ProbLoss` (`aspan_loss.py`) を参考）。
    * `L_{geom}`: **【新規】** 幾何ヘッドに対する補助損失。Mambaが処理の過程で幾何情報を失わないように制約をかける。

### 4.2. 評価指標とデータセット (Evaluation Metrics & Datasets)

* **評価指標**:
    * **精度**: Pose Estimation AUC @(5°, 10°, 20°), Homography Estimation Accuracy, etc.
    * **効率**: Latency (ms), FLOPs (G), Parameters (M)
* **データセット**:
    * **学習**: MegaDepth (屋外), ScanNet (室内) を中心とした混合データセット。
    * **評価**: 上記に加え、HPatches, YFCC100M, IMC 2022など、複数の標準ベンチマーク。

### 4.3. 実験計画 (Experiment Plan)

1.  **スキャン戦略の比較**: 多方向スキャン vs ヒルベルト曲線スキャン
2.  **ヘッドの有効性**: マルチヘッドMamba vs シングルヘッドMamba
3.  **幾何フィードバックの有効性**: 幾何ヘッドからのフィードバックの有無
4.  **Flow予測器の比較**: KAN vs MLP (ASpanFormerの元実装)

### 4.4. 潜在的リスクと対策 (Potential Risks & Mitigation)

* **リスク1**: マルチヘッドMambaが機能分離を学習しない。
    * **対策**: 補助損失の導入。シングルヘッドへのフォールバック計画。
* **リスク2**: ヒルベルト曲線の実装がボトルネックになる。
    * **対策**: 多方向スキャンをベースラインとして開発を並行。
* **リスク3**: 汎化性能が想定より低い。
    * **対策**: データ多様化戦略、データ依存パラメータへの正解則化。

---

## 5. 今後の進め方 (Next Steps)

### 5.1. 開発ロードマップ (案)

* **フェーズ1: 基盤構築とプロトタイピング (1〜3ヶ月)**
    * **目標**: 主要コンポーネントを実装し、最小構成でモデルが動作・学習することを確認する。
* **フェーズ2: 性能最大化と中核技術の深化 (3〜6ヶ月)**
    * **目標**: 中核となる革新技術（スキャン戦略、マルチヘッド）の比較実験を行い、最良の組み合わせでSOTA精度を目指す。
* **フェーズ3: 汎化性能の確立と論文執筆 (2〜3ヶ月)**
    * **目標**: 複数データセットでの学習と評価を通じてモデルの堅牢性を証明し、トップカンファレンスに投稿可能な成果をまとめる。

### 5.2. TODOリスト (フェーズ1)

- [ ] **環境構築**: PyTorch, CUDA環境のセットアップ
- [ ] **データローダーの実装**: MegaDepthデータセットのローダーを作成
- [ ] **アーキテクチャ実装**:
    - [ ] `Feature Encoder` (JamMaベース)
    - [ ] `Mamba Initializer` (Multi-Head版)
    - [ ] `AS-Mamba Block` の骨格
    - [ ] `KAN` の実装
    - [ ] `Local Mamba Module` (多方向スキャン版から)
    - [ ] `Matching Module` (JamMaベース)
- [ ] **学習パイプライン**:
    - [ ] 損失関数の実装
    - [ ] 学習ループを回し、動作確認