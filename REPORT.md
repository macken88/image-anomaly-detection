# 実験レポート — MVTec 画像異常検知（Mahalanobis / PaDiM）

## 目的と前提
- 目的: 画像データの異常検知タスクにおいて、複数手法（Mahalanobis 距離法 / PaDiM）および複数バックボーン（ResNet18 / EfficientNet‑B0）で性能を比較する。
- 前提: 異常データが得られない、あるいは正常に比べて極めて少数、または未知の異常（観測済みのどの異常にも分類不能）が発生する状況を想定。よって訓練には正常データのみを使用し、正常データの分布推定（共分散行列など）と OOF 正常スコアに基づく閾値設定（z 空間の 99% 分位点）を行った。

## データと実験条件
- データ: MVTec AD（カテゴリ例: carpet, leather, tile, bottle, cable, pill）
  - 選定理由: テスクチャ系で3カテゴリ（carpet, leather, tile）、オブジェクト系で3カテゴリ（bottle, cable, pill）を選定。
- バックボーン: `resnet18`, `efficientnet_b0`
- 手法: Mahalanobis, PaDiM
- 検証手順（experiment.ipynb 内 `run_pipeline`）
  - K‑Fold で train/good を分割し、fold ごとに学習・検証スコア取得
  - 全フォールドのOOF正常スコアを結合し正規化。その 99% 分位点を正常・異常の閾値 `z_threshold` として決定
  - 全 train/good で最終モデルを学習し、test でAUCおよびF1スコアを算出
  - 出力保存先: `results/dev/<timestamp>/<backbone>/<category>/<method>/`
  - 集計: `results/dev/<timestamp>/sweep_summary.csv`

## 手法の要点（実装）
- Mahalanobis: 学習正常特徴の平均・共分散（Ledoit‑Wolf 等）から精度行列を推定し距離をスコア化（`anomaly_detectors/mahalanobis.py`）。
- PaDiM: 指定層特徴をサブサンプル（d）し、パッチ分布を多変量正規でモデリング、画素スコアを画像スコアへ集約（`anomaly_detectors/padim.py`）。
- 閾値: OOF 正常スコアを z 化し 99% 分位点を保存。最終モデル側でも学習正常スコア（μ, σ）で z 化し、`z > z_threshold` を異常判定に適用。

## 結果サマリー（sweep_summary.csv より）
- 手法平均（全カテゴリ・全バックボーンの平均）
  - Mahalanobis: 平均 AUC ≈ 0.945、平均 F1 ≈ 0.848
  - PaDiM: 平均 AUC ≈ 0.913、平均 F1 ≈ 0.765
- バックボーン × 手法の平均（例）
  - EfficientNet‑B0 × Mahalanobis: AUC ≈ 0.967、F1 ≈ 0.848
  - EfficientNet‑B0 × PaDiM: AUC ≈ 0.921、F1 ≈ 0.773
  - ResNet18 × Mahalanobis: AUC ≈ 0.923、F1 ≈ 0.848
  - ResNet18 × PaDiM: AUC ≈ 0.904、F1 ≈ 0.756

※ 値は実行時の `results/dev/<timestamp>/sweep_summary.csv` を基にしています。

## カテゴリ別ハイライト（final F1 の最良構成）
- carpet: PaDiM × EfficientNet‑B0（F1 ≈ 0.977, AUC ≈ 0.998）
- leather: PaDiM × EfficientNet‑B0（F1 ≈ 0.979, AUC = 1.000）
- tile: PaDiM × EfficientNet‑B0（F1 ≈ 0.930, AUC ≈ 0.999）
- bottle: PaDiM × ResNet18（F1 ≈ 0.984, AUC ≈ 0.998）
- cable: Mahalanobis × ResNet18（F1 ≈ 0.760, AUC ≈ 0.931）
- pill: Mahalanobis × ResNet18（F1 ≈ 0.916, AUC ≈ 0.876）

## 可視化とアーティファクト
- スコア分布ヒストグラム: `<backbone>/<category>/<method>/fold_?_hist.png`
- PaDiM ヒートマップのサンプル保存（原画像＋ヒートマップ並置 PNG）: `<backbone>/<category>/PaDiM/final_heatmaps/` に good×2, 異常×3
- バックボーン比較スロープグラフ: ノート末尾セル（`final_f1` / `final_auc` の双方で比較可能）

## 考察
- PaDiM は carpet / leather / tile / bottle で非常に強く、特に EfficientNet‑B0 と組み合わせた場合に AUC / F1 とも高水準。
- cable / pill では Mahalanobis が優位。細線状や微小な異常では、PaDiM の層選択や d の設定がマッチしにくい可能性。
- バックボーン差は AUC では明確に出る一方、F1 は「単一点のしきい値」依存のため一致し得る。z 分位による一律の運用ポリシー（99%）が、カテゴリによっては同一の混同行列を生みやすいことがある。
- `z_threshold` は概ね 3〜4.7 の範囲。PaDiM はカテゴリにより学習正常スコアの分散が大きく、z 化による正規化はしているものの、外れ値の影響を被りやすいカテゴリがある。

## 改善案 / 次アクション
- PaDiM のカテゴリ別チューニング（cable / pill 重点）
  - 層選択を浅めに（エッジ・細部強調）、`d` を縮小して冗長次元を抑制
  - 画像スコア集約（max / 上位パーセンタイル）をカテゴリ特性に合わせて調整
- 閾値最適化の多様化
  - OOF 分位点（97/98/99%）のカテゴリ別スイープ
  - EVT（POT）や Youden 指標、Conformal 分位点など、低 FPR 制御・汎化性重視のしきい値設計
- 運用構成
  - PaDiM+EffB0 を carpet / leather / tile / bottle のデフォルトに、Mahalanobis を cable / pill の保守構成として併用
  - レポート用に PR‑AUC / ROC‑AUC を併記し、F1 とランク評価を両立

## 再現手順（要点）
1. ノート先頭の「データ準備（anomalib）」セルで MVTec を準備（`MVTEC_ROOT` 可）
2. 「スイープ設定と実行」で対象バックボーンとカテゴリを指定
3. 実行後、`results/dev/<timestamp>/sweep_summary.csv` と各サブフォルダを参照
4. スロープグラフセルでバックボーン差を可視化
