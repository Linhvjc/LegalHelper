from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class KidDatagArguments:
    data_dir: str = field(metadata={'help': 'Path to save, load model.'})
    benchmark_dir: str = field(metadata={'help': 'The benchmark data dir.'})
    token_level: str = field(
        default='word-level',
        metadata={
            'help': 'Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level].',
        },
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={
            'help': 'Toggle whether to drop the last incomplete batch in the dataloader.',
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={'help': 'Number of workers for the dataloader.'},
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={'help': 'Toggle whether to use pinned memory in the dataloader.'},
    )

    max_seq_len_query: int = field(
        default=64,
        metadata={
            'help': 'The maximum total input sequence length for query after tokenization.',
        },
    )
    max_seq_len_document: int = field(
        default=256,
        metadata={
            'help': 'The maximum total input sequence length for document after tokenization.',
        },
    )

    use_lowercase: bool = field(
        default=True,
        metadata={
            'help': 'Set to True to convert all text to lowercase. Set to False to keep text as-is.',
        },
    )
    use_remove_punc: bool = field(
        default=True,
        metadata={
            'help': 'Set to True to remove punctuation from text. Set to False to keep punctuation.',
        },
    )
