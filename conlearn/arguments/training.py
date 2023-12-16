from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal


@dataclass
class KidTrainingArguments:
    logging_steps: int = field(
        default=200,
        metadata={'help': 'Number of steps between each logging update.'},
    )
    eval_steps: int = field(
        default=200,
        metadata={'help': 'Number of steps between each model evaluation.'},
    )
    save_steps: int = field(
        default=200,
        metadata={'help': 'Number of steps between each checkpoint saving.'},
    )
    wandb_project: str = field(
        default='semantic-similarity',
        metadata={'help': 'Name of the Weight and Bias project.'},
    )
    wandb_run_name: str = field(
        default='kid-experiment',
        metadata={'help': 'Name of the run for Weight and Bias.'},
    )
    wandb_watch: str = field(
        default='false',
        metadata={
            'help': 'Whether to enable tracking of gradients and model topology in Weight and Bias.',
        },
    )
    wandb_log_model: str = field(
        default='false',
        metadata={'help': 'Whether to enable model versioning in Weight and Bias.'},
    )
    do_train: bool = field(
        default=False,
        metadata={'help': 'Flag indicating whether to run the training process.'},
    )
    do_eval: bool = field(
        default=False,
        metadata={
            'help': 'Flag indicating whether to perform evaluation on the test/eval set.',
        },
    )
    seed: int = field(
        default=42, metadata={
            'help': 'Random seed used for initialization.',
        },
    )
    num_train_epochs: float = field(
        default=2.0,
        metadata={'help': 'Total number of training epochs to perform.'},
    )
    train_batch_size: int = field(
        default=32, metadata={'help': 'Batch size used for training.'},
    )
    eval_batch_size: int = field(
        default=64, metadata={'help': 'Batch size used for evaluation.'},
    )
    no_cuda: bool = field(
        default=False,
        metadata={
            'help': 'Flag indicating whether to avoid using CUDA when available.',
        },
    )
    gpu_id: int = field(
        default=0, metadata={
            'help': 'ID of the GPU to be used for computation.',
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            'help': 'Enable gradient checkpointing to reduce memory usage during training. When this flag is set, intermediate activations are recomputed during backward pass, which can be memory-efficient but might increase training time.',
        },
    )
    optimizer: Literal['AdamW', '8bitAdam'] = field(
        default='AdamW',
        metadata={
            'help': 'Specify the optimizer to use (choices: AdamW, 8bitAdam).',
        },
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={'help': 'The initial learning rate for Adam.'},
    )
    lr_scheduler_type: str = field(
        default='cosine',
        metadata={
            'help': "Type of learning rate scheduler to use. Available options are: 'cosine', 'step', 'plateau'. The default is 'cosine', which uses a cosine annealing schedule.",
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={
            'help': 'Weight decay if we apply some.',
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            'help': 'Number of updates steps to accumulate before performing a backward/update pass.',
        },
    )
    adam_epsilon: float = field(
        default=1e-9, metadata={'help': 'Epsilon for Adam optimizer.'},
    )
    adam_beta1: float = field(
        default=0.9, metadata={
            'help': 'Beta1 for Adam optimizer.',
        },
    )
    adam_beta2: float = field(
        default=0.98, metadata={
            'help': 'Beta2 for Adam optimizer.',
        },
    )
    max_steps: int = field(
        default=-1,
        metadata={
            'help': 'If > 0: set total number of training steps to perform. Override num_train_epochs.',
        },
    )
    max_grad_norm: float = field(
        default=1.0, metadata={
            'help': 'Max gradient norm.',
        },
    )
    warmup_steps: int = field(
        default=100, metadata={
            'help': 'Linear warmup over warmup_steps.',
        },
    )

    early_stopping: int = field(
        default=50,
        metadata={
            'help': 'Number of unincreased validation step to wait for early stopping.',
        },
    )
    tuning_metric: str = field(
        default='loss', metadata={
            'help': 'Metrics to tune when training.',
        },
    )
