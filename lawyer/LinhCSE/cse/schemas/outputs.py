from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class DenseOutput(ModelOutput):
    total_loss: torch.FloatTensor | None = None
    loss_ct: torch.FloatTensor | None = None
    loss_ct_dpi: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
