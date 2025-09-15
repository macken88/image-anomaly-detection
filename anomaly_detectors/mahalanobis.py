from __future__ import annotations


from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .utils import default_device

try:
    from sklearn.covariance import ledoit_wolf as _ledoit_wolf

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def _flatten_global(feat: torch.Tensor) -> torch.Tensor:
    """特徴マップをベクトル化する（Global Pool/Flatten）。

    入力が `[N, C, H, W]` の場合は `[N, C*H*W]` に変換する。
    既に `[N, D]` の場合はそのまま返す。
    """
    if feat.dim() == 4:
        return feat.view(feat.size(0), -1)
    return feat


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

    device = default_device(device)

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
    """任意の `DataLoader` 内の全画像に対して Mahalanobis 距離を計算する。"""

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
