from __future__ import annotations

from typing import Optional

import torch


def default_device(device: Optional[torch.device] = None) -> torch.device:
    """CUDA が利用可能なら CUDA、なければ CPU を返す。"""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
