from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import torch

from cse.utils.utils import MODEL_CLASSES


@dataclass
class ModelArguments:
    model_dir: str = field(
        default='checkpoint', metadata={
            'help': 'Path to save, load model.',
        },
    )
    model_type: str = field(
        default='phobert',
        metadata={
            'help': 'Model type selected in the list: ' +
            ', '.join(MODEL_CLASSES.keys()),
        },
    )
    pretrained: bool = field(
        default=False,
        metadata={
            'help': 'Whether to initialize the model from a pretrained base model.',
        },
    )
    pretrained_path: str = field(
        default=None, metadata={
            'help': 'Path to the pretrained model.',
        },
    )
    resize_embedding_model: bool = field(
        default=False,
        metadata={
            'help': 'Resize model embedding model following length of vocab.',
        },
    )
    compute_dtype: torch.dtype = field(
        default=torch.float,
        metadata={
            'help': 'Used in quantization configs. Do not specify this argument manually.',
        },
    )
    pooler_type: str = field(
        default='cls',
        metadata={
            'help': 'What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).',
        },
    )
    sim_fn: str = field(
        default='dot',
        metadata={
            'help': "Similarity function to use for calculations (default: 'cosine').",
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            'help': 'Amount of label smoothing to apply (default: 0.0).',
        },
    )
    dpi_query: bool = field(
        default=False, metadata={
            'help': 'Flag to enable DPI query.',
        },
    )
    coff_dpi_query: float = field(
        default=0.1,
        metadata={
            'help': 'Coefficient for DPI query calculations (default: 0.1).',
        },
    )
    dpi_positive: bool = field(
        default=False, metadata={
            'help': 'Flag to enable DPI positive.',
        },
    )
    coff_dpi_positive: float = field(
        default=0.1,
        metadata={
            'help': 'Coefficient for DPI positive calculations (default: 0.1).',
        },
    )
    use_align_loss: bool = field(
        default=False, metadata={
            'help': 'Enable alignment loss mode.',
        },
    )
    coff_alignment: float = field(
        default=0.05,
        metadata={
            'help': 'Coefficient for alignment loss calculations (default: 0.05).',
        },
    )
    use_uniformity_loss: bool = field(
        default=False,
        metadata={'help': 'Enable uniformity loss mode.'},
    )
    coff_uniformity: float = field(
        default=0.05,
        metadata={
            'help': 'Coefficient for uniformity loss calculations (default: 0.05).',
        },
    )
