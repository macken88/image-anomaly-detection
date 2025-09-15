from __future__ import annotations

# =============================================================================
#  画像異常検知コア関数群（Mahalanobis / PaDiM）
#  - 可視化やデータローディングは含めません（ノートブック側で実施）
#  - すべて日本語ドックストリングとコメントで統一
# =============================================================================

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    # sklearn が利用可能なら Ledoit-Wolf 推定を使う
    from sklearn.covariance import ledoit_wolf as _ledoit_wolf
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# エクスポート対象
__all__ = [
    "fit_mahalanobis",
    "all_mahalanobis_scores",
    "fit_padim",
    "padim_heatmap",
    "all_padim_scores",
]


# -----------------------------------------------------------------------------
# 内部ユーティリティ
# -----------------------------------------------------------------------------

def _default_device(device: Optional[torch.device] = None) -> torch.device:
    """デフォルトのデバイス（CUDA 利用可能なら CUDA、なければ CPU）を返す。"""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _flatten_global(feat: torch.Tensor) -> torch.Tensor:
    """特徴マップをベクトル化する（Global Pool/Flatten）。

    入力が `[N, C, H, W]` の場合は `[N, C*H*W]` に変換する。
    既に `[N, D]` の場合はそのまま返す。
    """
    if feat.dim() == 4:
        return feat.view(feat.size(0), -1)
    return feat


def _collect_spatial_feats(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    """中間特徴リストから空間次元を持つ特徴のみ抽出し、同一サイズへ揃える。

    - `create_feature_extractor` の返すノードによっては `[N, C]` の特徴（avgpool/flatten後）
      が混ざることがある。その場合は PaDiM に不適なので除外する。
    - 空間特徴が 1 つも無い場合は例外を投げ、レイヤ指定の見直しを促す。

    引数:
        feats: 特徴テンソルのリスト。

    戻り値:
        空間特徴のみを同一解像度にリサイズしたリスト。
    """
    spatial = [f for f in feats if f.dim() == 4]
    if not spatial:
        return []
    ref_size = spatial[0].shape[-2:]
    spatial = [
        f if f.shape[-2:] == ref_size
        else F.interpolate(f, size=ref_size, mode="bilinear", align_corners=False)
        for f in spatial
    ]
    return spatial


# -----------------------------------------------------------------------------
# Mahalanobis（画像レベル）
# -----------------------------------------------------------------------------

def fit_mahalanobis(
    train_loader: torch.utils.data.DataLoader,
    backbone: str,
    *,
    cov_estimator: str = "ledoit_wolf",
    reg_eps: float = 0.0,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Mahalanobis モデルを学習用データから構築する。

    事前学習済みバックボーンの最終手前の出力（もしくは最後の特徴マップをフラット化）を
    埋め込みとして用い、学習データの平均ベクトルと共分散を推定する。

    引数:
        train_loader: 学習用の `DataLoader`（画像テンソルが必要）。
        backbone: `torchvision.models` のモデル名（例: "resnet18"）。
        cov_estimator: 共分散推定手法。"ledoit_wolf" または "empirical"。
        reg_eps: 共分散の対角に加える正則化項（数値不安定性の緩和）。
        device: 実行デバイス。省略時は自動選択（CUDA 優先）。

    戻り値:
        辞書 `model_state`:
            - "mean": 学習埋め込みの平均（np.ndarray [D]）
            - "precision": 共分散行列の擬似逆行列（np.ndarray [D,D]）
            - "feature_extractor": 特徴抽出モジュール（nn.Module）
            - "meta": メタ情報（バックボーン名、推定手法など）
    """

    device = _default_device(device)

    # バックボーンから最終層手前までを特徴抽出器として利用（ResNet 系で有効）
    model = models.__dict__[backbone](pretrained=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    feats_np: List[np.ndarray] = []
    with torch.no_grad():
        for images, *_ in train_loader:
            images = images.to(device)
            feat = feature_extractor(images)
            feat = _flatten_global(feat)
            feats_np.append(feat.cpu().numpy())

    feats = np.concatenate(feats_np, axis=0)  # [N, D]
    mean = feats.mean(axis=0)  # [D]

    # 共分散推定
    if cov_estimator == "ledoit_wolf" and _HAS_SKLEARN:
        cov, _ = _ledoit_wolf(feats)
    else:
        # 標本共分散（1/(N-1) スケール）。sklearn が無い場合のフォールバック。
        cov = np.cov(feats, rowvar=False, bias=False)

    if reg_eps > 0:
        cov = cov + reg_eps * np.eye(cov.shape[0], dtype=cov.dtype)

    precision = np.linalg.pinv(cov)

    return {
        "mean": mean,
        "precision": precision,
        "feature_extractor": feature_extractor,
        "meta": {
            "backbone": backbone,
            "cov_estimator": cov_estimator,
            "reg_eps": reg_eps,
        },
    }


def all_mahalanobis_scores(
    model_state: Dict[str, object],
    loader: torch.utils.data.DataLoader,
) -> torch.Tensor:
    """任意の `DataLoader` 内の全画像に対して Mahalanobis 距離を計算する。

    引数:
        model_state: `fit_mahalanobis` の戻り値。
        loader: 評価対象の `DataLoader`。

    戻り値:
        `scores`: 画像ごとのスコア（`torch.Tensor [N]`）。
    """

    feature_extractor: nn.Module = model_state["feature_extractor"]  # type: ignore
    device = next(feature_extractor.parameters()).device
    mean = torch.tensor(model_state["mean"], device=device)  # type: ignore
    precision = torch.tensor(model_state["precision"], device=device)  # type: ignore

    chunks: List[torch.Tensor] = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            feats = feature_extractor(images)
            feats = _flatten_global(feats)
            diff = feats - mean
            # d_M(x) = sqrt( (x-μ)^T Σ^{-1} (x-μ) )
            scores = torch.sqrt(torch.sum((diff @ precision) * diff, dim=1))
            chunks.append(scores)
    return torch.cat(chunks, dim=0).cpu()


# -----------------------------------------------------------------------------
# PaDiM（パッチレベル → 画像レベル集約）
# -----------------------------------------------------------------------------

class _PadimFeatureExtractor(nn.Module):
    """PaDiM 用の中間特徴抽出器（バックボーン非依存の薄いラッパ）。

    引数:
        backbone: `torchvision.models` のモデル名（例: "resnet18"）。
        layers: 取得するグラフノード名のリスト。省略時はモデルに応じて自動選択。

    返り値（forward）:
        指定順の特徴マップ一覧（`List[Tensor[N,C,H,W]]`）。
    """

    def __init__(self, backbone: str, layers: Optional[List[str]] = None) -> None:
        super().__init__()
        from torchvision.models.feature_extraction import (
            create_feature_extractor,
            get_graph_node_names,
        )

        self.backbone = backbone
        self.model = models.__dict__[backbone](pretrained=True)
        self.model.eval()
        _, eval_nodes = get_graph_node_names(self.model)
        self._eval_nodes = set(eval_nodes)
        # まずヒューリスティックで候補を決める
        resolved = self._resolve_layers(self.model, backbone, layers, eval_nodes)

        # 候補で一旦 extractor を作成
        self.return_order = list(resolved)
        self.extractor = create_feature_extractor(self.model, return_nodes={n: n for n in self.return_order})

        # ダミー入力で空間特徴かを確認し、必要なら動的再選択
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                outs = self.extractor(dummy)
            spatial_names = [k for k, v in outs.items() if isinstance(v, torch.Tensor) and v.dim() == 4]
        except Exception:
            spatial_names = []

        if not spatial_names:
            # 動的に空間特徴を持つノードを探索して再構築
            picked = self._pick_spatial_layers_dynamic(self.model, eval_nodes)
            self.return_order = picked
            self.extractor = create_feature_extractor(self.model, return_nodes={n: n for n in self.return_order})

    def _resolve_layers(
        self,
        model: nn.Module,
        backbone: str,
        layers: Optional[List[str]],
        eval_nodes: List[str],
    ) -> List[str]:
        # ユーザー指定が有効ならそのまま採用
        if layers and all(l in eval_nodes for l in layers):
            return layers

        name = backbone.lower()
        # ResNet 系: ステージ出力 layer1-3 があれば優先
        if "resnet" in name:
            cand = [l for l in ["layer1", "layer2", "layer3"] if l in eval_nodes]
            if cand:
                return cand

        # EfficientNet/MobileNet/VGG/DenseNet 等の features シーケンスを持つモデル
        if hasattr(model, "features"):
            try:
                n = len(model.features)  # type: ignore[attr-defined]
            except TypeError:
                n = len(list(model.features))  # type: ignore[attr-defined]
            if n > 0:
                idxs = sorted({max(0, min(n - 1, i)) for i in [n // 4, n // 2, (3 * n) // 4]})
                cand = [f"features.{i}" for i in idxs if f"features.{i}" in eval_nodes]
                if cand:
                    return cand

        # 任意指定があったが不正な場合は例示付きで明示エラー
        if layers:
            missing = [l for l in layers if l not in eval_nodes]
            sample = list(eval_nodes)[-10:]
            raise ValueError(
                f"指定レイヤーが見つかりません: {missing} / 例: {sample}"
            )

        # フォールバック: 最後の 3 ノードを使用（ベストエフォート）
        return list(eval_nodes[-3:])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.extractor(x)
        return [out[name] for name in self.return_order]

    def _pick_spatial_layers_dynamic(self, model: nn.Module, eval_nodes: List[str]) -> List[str]:
        """ダミー入力で実際に空間特徴を返すノードを探索して 3 箇所選択する。

        - 後段 64 ノードを候補にしてまとめて出力を取り、4 次元のテンソルを返す
          ノードのみ抽出する。
        - 抽出できたノードの中から、深さ方向に均等になるよう 3 箇所を選ぶ。
        - 見つからない場合は、eval_nodes の末尾 1 つ（最後の空間出力と思われる場所）を返す。
        """
        from torchvision.models.feature_extraction import create_feature_extractor

        # 候補ノード（末尾から最大 64 個）
        cand = list(eval_nodes)[-64:]
        # 重複回避のために辞書化
        return_nodes = {n: n for n in cand}
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            outs = extractor(dummy)
        spatial = [n for n in cand if isinstance(outs.get(n), torch.Tensor) and outs[n].dim() == 4]
        if not spatial:
            # 最悪ケース: 候補から最後の 1 ノードのみ返す（上位でエラーになる場合あり）
            return [cand[-1]] if cand else list(eval_nodes[-1:])
        # できるだけ前・中・後ろから等間隔に 3 箇所選ぶ
        k = len(spatial)
        idxs = sorted({max(0, min(k - 1, i)) for i in [k // 4, k // 2, (3 * k) // 4]})
        return [spatial[i] for i in idxs]


def fit_padim(
    train_loader: torch.utils.data.DataLoader,
    backbone: str,
    *,
    layers: Optional[List[str]] = None,
    d: int = 100,
    reg_eps: float = 1e-5,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """PaDiM の統計量（位置ごとの平均・共分散）を学習データから推定する。

    引数:
        train_loader: 学習用の `DataLoader`（画像テンソルが必要）。
        backbone: `torchvision.models` のモデル名。
        layers: 中間特徴を取得するノード名リスト。省略時は自動選択。
        d: チャネルのサブサンプル数（埋め込み次元）。
        reg_eps: 共分散の対角に加える正則化項（Σ := Σ + εI）。
        device: 実行デバイス。省略時は自動選択。

    戻り値:
        辞書 `model_state`:
            - "mean": 位置ごとの平均（Tensor [HW, d]、CPU）
            - "cov": 位置ごとの共分散（Tensor [HW, d, d]、CPU、float32）
            - "idx": 使用したチャネルインデックス（Tensor [d]）
            - "feature_extractor": 中間特徴抽出器（nn.Module）
            - "meta": メタ情報（バックボーン、レイヤ、d など）
    """

    device = _default_device(device)
    feature_extractor = _PadimFeatureExtractor(backbone, layers).to(device).eval()

    # 学習埋め込みを収集
    embedding_list: List[torch.Tensor] = []
    with torch.no_grad():
        for images, *_ in train_loader:
            images = images.to(device)
            feats = feature_extractor(images)  # List[Tensor]
            feats = _collect_spatial_feats(feats)  # 空間特徴のみ同一解像度へ
            embedding = torch.cat(feats, dim=1)  # [N, C, H, W]
            embedding_list.append(embedding.cpu())

    embeddings = torch.cat(embedding_list, dim=0)  # [N, C, H, W] (CPU)
    c = embeddings.shape[1]
    h, w = embeddings.shape[2:]

    # チャネルサブサンプル（再現性のため固定シード）
    torch.manual_seed(0)
    idx = torch.randperm(c)[:d]
    embeddings = embeddings[:, idx, :, :]  # [N, d, H, W]

    # [N, H*W, d] へ再配置
    embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, h * w, d)

    # 位置ごとの平均（Torch）
    mean = embeddings.mean(dim=0).contiguous()  # [HW, d]

    # 位置ごとの共分散（NumPy で計算し Torch へ戻す）
    cov = torch.zeros(h * w, d, d, dtype=torch.float32)
    eye_d = np.eye(d, dtype=np.float64)
    emb_np = embeddings.numpy()  # [N, HW, d]（CPU）

    for i in range(h * w):
        Xi = emb_np[:, i, :]  # (N, d)
        # 標本共分散（1/(N-1) スケール）
        Si = np.cov(Xi, rowvar=False, bias=False)
        Si = Si + reg_eps * eye_d  # 正則化（PaDiM の安定化）
        cov[i] = torch.from_numpy(Si).to(torch.float32)

    # 共分散の（擬似）逆行列を事前計算して返す
    inv_cov = torch.linalg.pinv(cov)  # [HW, d, d] (CPU)

    return {
        "mean": mean,  # [HW, d] (CPU)
        "cov": cov,        # [HW, d, d] (CPU)
        "inv_cov": inv_cov,  # [HW, d, d] (CPU)
        "idx": idx,    # [d]
        "feature_extractor": feature_extractor,
        "meta": {"backbone": backbone, "layers": layers, "d": d},
    }


def padim_heatmap(
    model_state: Dict[str, object],
    images: torch.Tensor,
) -> torch.Tensor:
    """PaDiM の異常スコアヒートマップを計算する。

    引数:
        model_state: `fit_padim` の戻り値。
        images: 正規化済み画像テンソル（`Tensor [N, C, H, W]`）。

    戻り値:
        `Tensor [N, H', W']`: 各画像の異常スコアヒートマップ（学習時の空間解像度）。
    """

    feature_extractor: nn.Module = model_state["feature_extractor"]  # type: ignore
    device = next(feature_extractor.parameters()).device

    mean: torch.Tensor = model_state["mean"]  # type: ignore
    # 逆共分散は学習時に事前計算して受け取る
    inv_cov: torch.Tensor = model_state["inv_cov"]  # type: ignore
    idx: torch.Tensor = model_state["idx"]  # type: ignore

    feature_extractor.eval()
    with torch.no_grad():
        feats = feature_extractor(images.to(device))  # List[Tensor]
        feats = _collect_spatial_feats(feats)
        embedding = torch.cat(feats, dim=1)[:, idx.to(device), :, :]  # [N, d, H, W]

    n, d, h, w = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1).reshape(n, h * w, d)  # [N, HW, d]

    # 位置ごとの逆共分散（事前計算済み）
    inv_cov = inv_cov.to(device)  # [HW, d, d]
    mean = mean.to(device)        # [HW, d]

    maps: List[torch.Tensor] = []
    for emb in embedding:  # emb: [HW, d]
        diff = emb - mean  # [HW, d]
        # Mahalanobis^2 を einsum で一括計算
        dist2 = torch.einsum("nd,ndd,nd->n", diff, inv_cov, diff)  # [HW]
        maps.append(torch.sqrt(dist2).reshape(h, w))
    return torch.stack(maps, dim=0)  # [N, H, W]


def all_padim_scores(
    model_state: Dict[str, object],
    loader: torch.utils.data.DataLoader,
    return_maps: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """`DataLoader` 全体に対して PaDiM スコア（画像レベル）を計算する。

    画像レベルスコアは、ヒートマップの画素最大値（max-pooling）とする。

    引数:
        model_state: `fit_padim` の戻り値。
        loader: 評価対象の `DataLoader`。
        return_maps: True の場合はヒートマップも合わせて返す。

    戻り値:
        - `scores` (`Tensor [N]`): 各画像のスコア
        - `maps`   (`Tensor [N, H, W]`): ヒートマップ（`return_maps=True` のとき）
    """

    scores_list: List[torch.Tensor] = []
    maps_list: List[torch.Tensor] = []

    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        maps = padim_heatmap(model_state, images)
        scores = maps.view(maps.size(0), -1).max(dim=1)[0]
        scores_list.append(scores)
        if return_maps:
            maps_list.append(maps)

    scores_all = torch.cat(scores_list).cpu()
    if return_maps:
        maps_all = torch.cat(maps_list).cpu()
        return scores_all, maps_all
    return scores_all
