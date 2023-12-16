from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class KidDenseOutput(ModelOutput):
    total_loss: torch.FloatTensor | None = None
    loss_ct: torch.FloatTensor | None = None
    loss_ct_dpi: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None


@dataclass
class KidSparseOutput(ModelOutput):
    total_loss: torch.FloatTensor | None = None
    loss_ct: torch.FloatTensor | None = None
    loss_ct_dpi_query: torch.FloatTensor | None = None
    loss_ct_dpi_positive: torch.FloatTensor = None


@dataclass
class MocoOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    accuracy: float | None = None
    std_query: float | None = None
    std_key: float | None = None
