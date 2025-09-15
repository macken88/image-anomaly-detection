from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .utils import default_device


def _collect_spatial_feats(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    """中間特徴リストから空間次元を持つ特徴のみ抽出し、同一サイズへ揃える。"""
    spatial = [f for f in feats if f.dim() == 4]
    if not spatial:
        return []
    ref_size = spatial[0].shape[-2:]
    spatial = [
        (
            f
            if f.shape[-2:] == ref_size
            else F.interpolate(f, size=ref_size, mode="bilinear", align_corners=False)
        )
        for f in spatial
    ]
    return spatial


class _PadimFeatureExtractor(nn.Module):
    """PaDiM 用の中間特徴抽出器（バックボーン非依存の薄いラッパ）。"""

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
        resolved = self._resolve_layers(self.model, backbone, layers, eval_nodes)

        self.return_order = list(resolved)
        self.extractor = create_feature_extractor(
            self.model, return_nodes={n: n for n in self.return_order}
        )

        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                outs = self.extractor(dummy)
            spatial_names = [
                k
                for k, v in outs.items()
                if isinstance(v, torch.Tensor) and v.dim() == 4
            ]
        except Exception:
            spatial_names = []

        if not spatial_names:
            picked = self._pick_spatial_layers_dynamic(self.model, eval_nodes)
            self.return_order = picked
            self.extractor = create_feature_extractor(
                self.model, return_nodes={n: n for n in self.return_order}
            )

    def _resolve_layers(
        self,
        model: nn.Module,
        backbone: str,
        layers: Optional[List[str]],
        eval_nodes: List[str],
    ) -> List[str]:
        if layers and all(l in eval_nodes for l in layers):
            return layers

        name = backbone.lower()
        if "resnet" in name:
            cand = [l for l in ["layer1", "layer2", "layer3"] if l in eval_nodes]
            if cand:
                return cand

        if hasattr(model, "features"):
            try:
                n = len(model.features)  # type: ignore[attr-defined]
            except TypeError:
                n = len(list(model.features))  # type: ignore[attr-defined]
            if n > 0:
                idxs = sorted(
                    {max(0, min(n - 1, i)) for i in [n // 4, n // 2, (3 * n) // 4]}
                )
                cand = [f"features.{i}" for i in idxs if f"features.{i}" in eval_nodes]
                if cand:
                    return cand

        if layers:
            missing = [l for l in layers if l not in eval_nodes]
            sample = list(eval_nodes)[-10:]
            raise ValueError(f"指定レイヤーが見つかりません: {missing} / 例: {sample}")

        return list(eval_nodes[-3:])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.extractor(x)
        return [out[name] for name in self.return_order]

    def _pick_spatial_layers_dynamic(
        self, model: nn.Module, eval_nodes: List[str]
    ) -> List[str]:
        """ダミー入力で実際に空間特徴を返すノードを探索して 3 箇所選択する。"""
        from torchvision.models.feature_extraction import create_feature_extractor

        cand = list(eval_nodes)[-64:]
        return_nodes = {n: n for n in cand}
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            outs = extractor(dummy)
        spatial = [
            n
            for n in cand
            if isinstance(outs.get(n), torch.Tensor) and outs[n].dim() == 4
        ]
        if not spatial:
            return [cand[-1]] if cand else list(eval_nodes[-1:])
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
    """PaDiM の統計量（位置ごとの平均・共分散）を学習データから推定する。"""

    device = default_device(device)
    feature_extractor = _PadimFeatureExtractor(backbone, layers).to(device).eval()

    embedding_list: List[torch.Tensor] = []
    with torch.no_grad():
        for images, *_ in train_loader:
            images = images.to(device)
            feats = feature_extractor(images)
            feats = _collect_spatial_feats(feats)
            embedding = torch.cat(feats, dim=1)
            embedding_list.append(embedding.cpu())

    embeddings = torch.cat(embedding_list, dim=0)
    c = embeddings.shape[1]
    h, w = embeddings.shape[2:]

    torch.manual_seed(0)
    idx = torch.randperm(c)[:d]
    embeddings = embeddings[:, idx, :, :]

    embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, h * w, d)
    mean = embeddings.mean(dim=0).contiguous()

    cov = torch.zeros(h * w, d, d, dtype=torch.float32)
    eye_d = np.eye(d, dtype=np.float64)
    emb_np = embeddings.numpy()

    for i in range(h * w):
        Xi = emb_np[:, i, :]
        Si = np.cov(Xi, rowvar=False, bias=False)
        Si = Si + reg_eps * eye_d
        cov[i] = torch.from_numpy(Si).to(torch.float32)

    inv_cov = torch.linalg.pinv(cov)

    return {
        "mean": mean,
        "cov": cov,
        "inv_cov": inv_cov,
        "idx": idx,
        "feature_extractor": feature_extractor,
        "meta": {"backbone": backbone, "layers": layers, "d": d},
    }


def padim_heatmap(
    model_state: Dict[str, object],
    images: torch.Tensor,
) -> torch.Tensor:
    """PaDiM の異常スコアヒートマップを計算する。"""

    feature_extractor: nn.Module = model_state["feature_extractor"]  # type: ignore
    device = next(feature_extractor.parameters()).device

    mean: torch.Tensor = model_state["mean"]  # type: ignore
    inv_cov: torch.Tensor = model_state["inv_cov"]  # type: ignore
    idx: torch.Tensor = model_state["idx"]  # type: ignore

    feature_extractor.eval()
    with torch.no_grad():
        feats = feature_extractor(images.to(device))
        feats = _collect_spatial_feats(feats)
        embedding = torch.cat(feats, dim=1)[:, idx.to(device), :, :]

    n, d, h, w = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1).reshape(n, h * w, d)

    inv_cov = inv_cov.to(device)
    mean = mean.to(device)

    maps: List[torch.Tensor] = []
    for emb in embedding:
        diff = emb - mean
        dist2 = torch.einsum("nd,ndd,nd->n", diff, inv_cov, diff)
        maps.append(torch.sqrt(dist2).reshape(h, w))
    return torch.stack(maps, dim=0)


def all_padim_scores(
    model_state: Dict[str, object],
    loader: torch.utils.data.DataLoader,
    return_maps: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """`DataLoader` 全体に対して PaDiM スコア（画像レベル）を計算する。"""

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
