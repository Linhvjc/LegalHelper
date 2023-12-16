from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

import torch
from retriever.dataset.featuring import convert_text_to_features
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class AbstractDataset(ABC, Dataset):
    """
    An abstract class for PyTorch datasets. It defines the structure
    and required methods that must be implemented by any subclass.
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialize the dataset with the path to the data.

        Parameters:
            data_path (Path): The path to the dataset file or directory.
        """
        self.data_path = data_path

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        raise NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Abstract method to get a data sample given an index.

        Parameters:
            idx (int): The index of the data sample to retrieve.

        Returns:
            torch.Tensor: The data sample corresponding to the given index.
        """
        raise NotImplementedError('Subclasses should implement this method')


class KidDataset(AbstractDataset):
    def __init__(self, args, data, tokenizer: PreTrainedTokenizer) -> None:
        self.args = args

        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, index: int):
        # preprocessing data
        data_point = self.data[index]

        # 1. query
        query = data_point['query']
        # 2. document # Suggest for Text augmentation
        document = data_point['document']

        input_ids_query, attention_mask_query = convert_text_to_features(
            text=query,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len_query,
            lower_case=self.args.use_lowercase,
            remove_punc=self.args.use_remove_punc,
        )
        input_ids_document, attention_mask_document = convert_text_to_features(
            text=document,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len_document,
            lower_case=self.args.use_lowercase,
            remove_punc=self.args.use_remove_punc,
        )

        return (
            torch.tensor(input_ids_query, dtype=torch.long),
            torch.tensor(attention_mask_query, dtype=torch.long),
            torch.tensor(input_ids_document, dtype=torch.long),
            torch.tensor(attention_mask_document, dtype=torch.long),
        )
