# 画像異常検知ポートフォリオ（MVTec / Mahalanobis・PaDiM）

本リポジトリは、製造業などの外観検査を想定した画像異常検知の再現性重視ポートフォリオです。開発用ノートブックでパイプライン設計を確定し、その固定設定を評価用ノートブックで複数カテゴリに一括適用します。主な対象手法は Mahalanobis 距離ベース法と PaDiM です。

Executive Summary:
- 設計確定は `01_experiments_dev.ipynb`、評価は `02_evaluation_report.ipynb` で一発実行。
- 固定パイプラインは `assets/fixed_pipeline.json` に保存し、リークを防止。
- 指標・図表は `runs/` および `assets/figs/` に保存し、再現可能に管理。


**目次**
- リポジトリ構成
- セットアップ（uv / pip）
- データ準備（MVTec AD）
- 使い方（開発→固定→評価）
- 成果物とレポート
- 再現性とルール
- トラブルシュート
- 謝辞


## リポジトリ構成

```
.
├── 01_experiments_dev.ipynb      # devカテゴリのみで設計確定（EDA/試行錯誤）
├── 02_evaluation_report.ipynb    # 固定設定でevalカテゴリを一発評価
├── assets/
│   ├── figs/                     # 代表図（小容量のみ格納）
│   └── fixed_pipeline.json       # ← 01で出力（無ければ02は実行しない想定）
├── runs/                         # 02の評価結果（metrics.json など）
├── datasets/MVTecAD/             # 例：既存のMVTec配置（任意）
├── MVtec_dataset/                # 例：既存のMVTec配置（任意）
├── pyproject.toml                # 依存関係（anomalib, torch ほか）
└── uv.lock                       # uv 用ロックファイル
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


## データ準備（MVTec AD）

- 基本方針: データ取得は anomalib の機能を用いてノートブック内から行います（直リンクや手動DLは推奨しません）。
- 既にローカルにデータがある場合は、環境変数 `MVTEC_ROOT` でルートを指定するか、ノートブック先頭セルの設定でパスを変更してください。
- 既存配置の例: `datasets/MVTecAD/` または `MVtec_dataset/`（いずれも本リポジトリに含めないことを推奨）。

ノートブック側の想定処理（擬似コード）:

```python
from pathlib import Path
import os

# 既存ならそのまま利用。未存在なら anomalib でダウンロード/展開。
MVTEC_ROOT = Path(os.environ.get("MVTEC_ROOT", "datasets/MVTecAD"))
MVTEC_ROOT.mkdir(parents=True, exist_ok=True)

# 例: anomalib のデータユーティリティ/Datamodule等で download=True を利用
# from anomalib.data import MVTec
# _ = MVTec(root=str(MVTEC_ROOT), category="leather", task="segmentation", download=True)
```


## 使い方（開発→固定→評価）

1) 開発（設計確定）: `01_experiments_dev.ipynb`
- devカテゴリ（例: `carpet`）のみを用いて、特徴抽出層やハイパラを試行。
- Mahalanobis：平均・共分散の推定（例：Ledoit–Wolf 縮小）と画像スコア化。
- PaDiM：使用層とチャネルサブサンプル `d` の選定。
- 閾値ルール：devのtestを用いて「画像レベル FPR = 1%」となるスコアを決定。
- 決定内容を `assets/fixed_pipeline.json` に保存（ノートブックが出力）。

2) 固定設定で評価: `02_evaluation_report.ipynb`
- 1) の JSON を読み込み、evalカテゴリ（例: `leather`, `tile`）に対して、`train/good` のみで fit → `test` を一度だけ評価。
- `runs/eval/**/metrics.json` に AUROC/AUPRC/F1 など、`assets/figs/**` に図を保存。


## 成果物とレポート

- 設定: `assets/fixed_pipeline.json`
- 指標: `runs/eval/<category>/<method>/metrics.json`（例：AUROC, AUPRC, F1）
- 図表: `assets/figs/`（ROC/PR、失敗例ギャラリー等）

README への反映テンプレート（例）:

| Category | Method       | AUROC | AUPRC | F1 (FPR=1%) |
|----------|--------------|------:|------:|------------:|
| leather  | Mahalanobis  | 0.98  | 0.97  | 0.82        |
| leather  | PaDiM        | 0.99  | 0.98  | 0.85        |
| tile     | Mahalanobis  | 0.97  | 0.96  | 0.80        |
| tile     | PaDiM        | 0.98  | 0.97  | 0.83        |

（実数値は `runs/**/metrics.json` の結果で更新してください）


## 再現性とルール

- 固定設定での評価時にパラメータ変更を行わない（データリーク防止）。
- 閾値は dev の test を起点に「画像FPR=1%」を満たすルール/値で固定。
- 乱数 `seed` は少なくとも複数（例: 0,1,2）で平均化。可能ならCIも併記。


## トラブルシュート

- PyTorch のインストールに失敗する: GPU/OSに適したホイールを選択してください。`pip install torch torchvision` の代わりに公式推奨コマンドを使用。
- ノートブックで MVTec のパスが見つからない: `MVTEC_ROOT` を設定、または先頭セルのパス設定を修正。
- 大容量成果物の管理: 画像は `assets/figs/` に小容量で保存し、巨大ファイルは追跡しない方針。


## 謝辞

- 本実装は [anomalib](https://github.com/openvinotoolkit/anomalib) と MVTec AD データセットに深く依拠しています。素晴らしいコミュニティと研究に感謝します。


---

補足（エージェント向け）:
- `assets/fixed_pipeline.json` が無い場合は評価（02）を実行せず、先に開発（01）を完了してください。
- 返答・コメントは日本語推奨。データ取得は anomalib 経由を原則とします。
