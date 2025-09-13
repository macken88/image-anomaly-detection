# AGENTS.md（Codex CLI 用 / ポートフォリオ：MVTec 画像異常検知）

> 本リポジトリは、製造業における **画像異常検知の汎用パイプライン** を示すポートフォリオです。 Mahalanobis 距離ベース法と PaDiM を対象に、**開発（設計確定）→評価（固定パイプライン適用）** の二段構成で検証します。
>
> **Codex CLI 等のエージェント**が、セットアップから評価・レポート作成までを安全に自動実行できるよう、役割・手順・品質基準を明文化します。

---

## 0. ゴール（WHAT）

- **固定パイプラインの再現**：`01_experiments_dev.ipynb` で設計を確定し、`assets/fixed_pipeline.json` に保存。
- **一発評価の実行**：`02_evaluation_report.ipynb` で **固定**パイプラインを **eval カテゴリ**（例：`leather`, `tile`）へ適用し、指標・図表を出力。
- **成果物**：
  - `assets/fixed_pipeline.json`（手法別設定＋閾値ルール）
  - `runs/eval/**/metrics.json`（AUROC/AUPRC/F1 など）
  - `assets/figs/**`（ROC/PR/失敗例ギャラリー）
  - `README.md` のサマリー表更新

---

## 1. 役割と行動規範（HOW）

### Developer（開発）

- `01_experiments_dev.ipynb` で **dev カテゴリ（例：carpet）****のみを使って、以下を****決定**し JSON に保存：
  - 特徴抽出層（ImageNet 事前学習モデルの指定など）、ハイパーパラメータの試行錯誤用・EDAも含む
  - 画像の前処理は固定
  - Mahalanobis：μ・Σ の **推定法**（例：Ledoit–Wolf 縮小）
  - PaDiM：使用層とチャネルサブサンプル `d`
  - 学習済みモデルはMahalanobis、PaDiMで共通
  - **閾値決定ルール**：dev の **test** で **画像レベル FPR=1%** となるスコア（数値 or ルール）
- **禁止**：eval の test 結果を見て設定・閾値を変更しない（リーク防止）。

### Evaluator（評価）

- `02_evaluation_report.ipynb` にて、**固定**設定を読み込み、eval カテゴリの **train\_normal でのみ fit** → **test を一度だけ評価**。
- 指標：AUROC / AUPRC /（固定閾値で）TPR・FPR・F1、可能なら seed 平均と CI。

### Reporter（報告）

- `README.md` の **Executive Summary** を 3 行で更新。
- 主要表（カテゴリ × 手法）と代表図（棒グラフ）を `assets/figs/` に保存し README から相対リンク。

---

## 2. リポジトリ規約（SCOPE）

```
.
├── README.md
├── requirements.txt              # 依存（最小）
├── 01_experiments_dev.ipynb      # 試行錯誤＆設計確定（dev のみ）
├── 02_evaluation_report.ipynb    # 固定パイプラインで eval を一発評価
├── assets/
│   ├── fixed_pipeline.json       # 01 が出力・02 が参照
│   └── figs/                     # 02 が保存（小容量画像のみ）
├── runs/                         # 指標・ログ（JSON/CSV/画像）
└── data/                         # MVTec 配置（Git 無管理）
```

- **データは Git に含めない**。`data/mvtec/` 直下に配置、または `MVTEC_ROOT` 環境変数でルートを指定。
- 生成物（`runs/**`, `assets/figs/**`）は差分確認のためコミット可（容量注意）。

---

## 3. セットアップ（SETUP）

- 推奨：Python 3.10+（GPU 環境があれば PyTorch を環境に合わせてインストール）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **データ取得は anomalib を用いて実行**（次節参照）。

---

## 4. データ取得（DATA ACQUISITION with anomalib）

**方針**：MVTec AD のダウンロード・配置は **anomalib 経由**で行う。手動の `wget` や外部スクリプトでの取得は行わない。

### 4.1 ノートブックからの取得（推奨）

- `01_experiments_dev.ipynb` および `02_evaluation_report.ipynb` の冒頭セルに、**anomalib を利用した取得ロジック**を実装：
  - 既存パス `MVTEC_ROOT` or `data/mvtec` を自動検出
  - 未検出なら anomalib の API / ユーティリティで **ダウンロード & 展開**
  - 完了後、カテゴリ構成を検証（`train/good`, `test/good|<defect>`）

> 例（擬似コード。anomalib のバージョンに応じて適宜 API を選択）：

```python
from pathlib import Path
import os

MVTEC_ROOT = Path(os.environ.get("MVTEC_ROOT", "data/mvtec"))
MVTEC_ROOT.mkdir(parents=True, exist_ok=True)

# anomalib によるダウンロード（download=True 等の公式手段を用いる）
# 例: datamodule / ユーティリティ経由（バージョンに応じた推奨APIを利用）
# from anomalib.data import MVTec
# _ = MVTec(root=str(MVTEC_ROOT), category="leather", task="segmentation", download=True)

assert MVTEC_ROOT.exists(), "MVTec root not found after anomalib download."
```

### 4.2 エージェント行動規範（データ）

- **必ず anomalib 経由で取得・展開**すること。
- **直リンク/手動 DL 禁止**。`data/mvtec` の存在を検知し、なければ anomalib を使って取得。
- 取得済みの場合は **再取得しない**（冪等）。

---

## 5. タスクの実行（TASKS）

### T1: 設計確定（dev）

1. `01_experiments_dev.ipynb` を **上から順に実行**。
2．手法：Mahalanobis
2-1．画像から特徴量を抽出（抽出層を試行錯誤できるように関数化）
2-2．トレーニングデータでCVを行い、各フォールドで訓練データ、訓練外データにおけるMahalanobis距離をヒストグラムで可視化
2-3．テストデータを使い、閾値と評価指標（AUROC、F1スコア）の関係性を可視化
3．手法：PaDiM
3-1．画像から特徴量を抽出（使用層を試行錯誤できるように関数化）
3-2．トレーニングデータでCVを行い、各フォールドで訓練データ、訓練外データにおけるヒートマップ最大値をヒストグラムで可視化
3-3．テストデータを使い、閾値と評価指標（AUROC、F1スコア）の関係性を可視化
3-4．いくつかのテスト画像に対し異常ヒートマップを表示
4. 手法：Mahalanobis / PaDiM の比較（seed = 0,1,2 の平均で良い）。
5. **決定事項**を `assets/fixed_pipeline.json` に保存（ノートが自動出力）。
   - 例：

```json
{
  "common": {"image_size": 256, "seeds": [0,1,2]},
  "threshold": {"image_fpr_target": 0.01, "source": "dev_carpet_test"},
  "mahalanobis": {"backbone": "resnet18", "cov_estimator": "ledoit_wolf"},
  "padim": {"layers": ["layer2","layer3"], "channel_subsample": 100}
}
```

### T2: 評価（eval 一発）

1. `02_evaluation_report.ipynb` を実行し、固定 JSON を読み込む。
2. eval カテゴリ（デフォルト：`leather`, `tile`）を選び、**fit は train\_normal のみ** → test を **一度だけ** 評価。
3. `runs/eval/**/metrics.json` と `assets/figs/**` を更新。README の表も自動/半自動で更新。

### T3: 追加タスク例（エージェント指示フォーマット）

- **「PaDiM の **``**=100→150 で dev 再評価と差分表作成」**
  - 01 に `d=150` 分岐を追加 → 同一 seed 群で再評価 → AUROC 差/CI のミニ表を出力。
- **「eval に **``** を追加」**
  - 02 のカテゴリ選択 UI に `wood` を追加 → 既存 JSON で評価 → 表と図を更新。

---

## 6. 品質ゲート（QUALITY GATES）

- ノートブックは **最後までエラーなく実行**。
- **リーク禁止**：02 でのパラメータ変更・閾値再学習を行わない。
- 乱数 `seeds=[0,1,2]` の平均と、可能ならブートストラップ CI を併記。
- 画像可視化は **同一スケール（色レンジ）ルール**を維持。

---

## 7. スタイルとコーディング規約（STYLE）

- Python セル：`black`（line length 88）/ `ruff` 推奨（任意）。
- 変数命名：`snake_case`、関数は docstring（要約一行＋引数/戻り値）。
- ノート構成：章立て（Header → Data → Methods → Results → Save JSON/Artifacts）。
- コメント：関数には使い方がわかるように、引数、返り値の型や、必須の情報（例：データフレームであれば期待されるカラム名など）を日本語で記載

---

## 8. セキュリティ / プライバシ（SECURITY）

- 外部秘匿情報は扱わない。API キー不要。
- 大容量データはリポジトリに追加しない（`.gitignore` 済み）。
- 依存は `requirements.txt` のみからインストール。

---

## 9. 既知の制約（LIMITATIONS）

- 評価は **MVTec の一部カテゴリ**に限定。
- 詳細な統計検定は任意（時間と GPU の有無に依存）。
- ノート以外の Python モジュール化は最小限（採用者の可読性優先）。

---

## 10. For Agents（実行時の注意）

- `AGENTS.md` を最初に読み、**anomalib によるデータ取得**と **リーク防止ルール**を順守すること。
- 返答、およびコメントは日本語とすること。
- `assets/fixed_pipeline.json` が存在しない場合は **02 を実行せず停止**し、01 の実行を促すこと。
- `threshold.source` が `dev_*_test` 以外なら **警告**（リーク疑い）。

