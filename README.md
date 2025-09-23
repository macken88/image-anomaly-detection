# 画像異常検知ポートフォリオ（MVTec / Mahalanobis・PaDiM）

本リポジトリは、製造業などの外観検査を想定した画像異常検知実験を実施したポートフォリオです。複数バックボーン × 複数カテゴリ × 複数手法（Mahalanobis / PaDiM）のスイープ実験を実行しました。

概要:
- `experiment.ipynb` 1本でデータ準備→CV→OOF z閾値→最終モデル→レポート用集計まで完了。
- 結果は `results/dev/<timestamp>/<backbone>/<category>/<method>/` に時刻付きで保存。
- PaDiM ヒートマップは各カテゴリから「good×2 + 異常×3」を原画像と横並びでPNG保存。
- バックボーン比較用のスロープチャート等、簡易可視化も内蔵。


**目次**
- リポジトリ構成
- セットアップ（uv / pip）
- データ準備（anomalib）
- 実行方法（experiment.ipynb）
- 出力物（フォルダ構成・ファイル）
- 実装の要点
- トラブルシュート
- 参考文献


## リポジトリ構成

```
.
├── experiment.ipynb                # 実験実行用ノートブック
├── anomaly_detectors/              # 手法の実装（Mahalanobis / PaDiM など）
├── datasets/MVTecAD/               # 例：既存のMVTec配置（任意、git管理外推奨）
└── results/                        # 実行結果（時刻別に保存）
```


## セットアップ（uv / pip）

前提:
- Python `>= 3.12`
- GPU は任意（CPUでも可）。GPU利用時は環境に合う PyTorch を選択してください。

### 方法A: uv（推奨）

```
# uv が未インストールの場合（任意）
pip install uv

# 仮想環境の作成と依存解決（.venv を自動管理）
uv sync

# Jupyter を起動（必要に応じて）
uvx jupyter lab  # または: uvx jupyter notebook
```

### 方法B: pip / venv

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .           # pyproject.toml の依存を解決

# Jupyter が未導入なら
pip install jupyterlab  # または: jupyter
```


## データ準備（anomalib）

- `experiment.ipynb` には anomalib を用いたデータダウンロードセル（存在しなければ取得）が含まれています。
- 既にローカルにデータがある場合は、環境変数 `MVTEC_ROOT` でルートを指定するか、先頭セルの設定を変更してください。
- 既存配置の例: `datasets/MVTecAD/` 


## 実行方法（experiment.ipynb）

1. ノートブックを開き、先頭の「共通設定」セルを実行（`device`・`PIN_MEMORY` などが自動設定）。
2. 「スイープ設定と実行」セルで `TARGET_BACKBONES` と `TARGET_CATEGORIES` を編集。
3. 実行すると以下を自動で行います。
   - K-Fold CV（train/good のみ）→ foldごとにヒストグラム保存。
   - Out Of Fold（OOF）の正常スコアから z 空間で 99% 分位点（z_threshold）を決定。
   - 全学習データ（train/good）で最終モデルをfitし、testを評価（AUC/F1）。
   - PaDiM は test のヒートマップから「good×2 + 異常×3」を抽出し、原画像と並置PNGを保存。
   - すべての結果を `results/dev/<timestamp>/...` に保存し、`sweep_summary.csv` を生成。
   - バックボーン間の比較のため、スロープグラフを表示。


## 出力物（フォルダ構成・ファイル）

```
results/dev/<timestamp>/
├── sweep_summary.csv                         # バックボーン×カテゴリ×手法の集計（AUC/F1, z_threshold 等）
└── <backbone>/<category>/<method>/
    ├── summary.csv                           # CV 各foldのメトリクス
    ├── fold_?_hist.png                       # スコア分布ヒストグラム
    └── final_heatmaps/                       # PaDiM: 原画像+熱図の横並びPNG（good×2, 異常×3）
```

## 実装の要点

- Mahalanobis：学習データの特徴埋め込み（平均・共分散）から精度行列を推定（Ledoit–Wolf など）し、距離をスコア化。
  - 計算用モジュール: `anomaly_detectors/mahalanobis.py`
    - `fit_mahalanobis(train_loader, backbone, feature_node=..., ...)` 学習用正常画像から平均ベクトル・共分散（精度行列）を推定。
    - `all_mahalanobis_scores(model_state, loader)` 任意のデータに対しマハラノビス距離を一括計算。
- PaDiM：指定層の特徴をサブサンプル（`d`）して多変量正規分布を仮定、画素スコアを画像スコアへ集約。
  - 計算用モジュール: `anomaly_detectors/padim.py`
    - `fit_padim(train_loader, backbone, layers=[...], d=...)` 学習用正常画像から指定層のパッチ分布を学習。
    - `all_padim_scores(model_state, loader, return_maps=False)` 画像の異常度スコア（必要に応じてヒートマップ）を取得。
- 閾値設計：
  - CV の各fold val（正常のみ）スコアを OOF として集約し、z空間に正規化して 99% 分位点 `z_threshold` を決定。
  - 最終モデルでも学習正常スコア（mu_final, sigma_final）で z 化し、`z > z_threshold` を異常判定に適用。
- DataLoader：`pin_memory` は GPU のとき True / CPU のとき False を自動設定。


## トラブルシュート

- anomalib が見つからない / ダウンロードに失敗する
  - ノートブックのデータ取得セルは anomalib の有無を判定して実行します。事前に `pip install anomalib` し、`MVTEC_ROOT` の書き込み権限をご確認ください。

- VRAM 不足で落ちる
  - `DEFAULT_BATCH_SIZE` や PaDiM の `d` を下げてください。`NUM_WORKERS` も環境に合わせて調整可能です。


## 参考文献
1) Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection [arXiv:2005.14140] : https://arxiv.org/abs/2005.14140
2) PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization [arXiv:2011.08785] : https://arxiv.org/abs/2011.08785
