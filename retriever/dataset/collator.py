from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class AbstractDataCollator(ABC):
    """
    An abstract class for a PyTorch DataCollator. It defines the structure
    and the required method that must be implemented by any subclass.
    """

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Abstract method that collates a batch of data samples into a format
        suitable for model training or evaluation.

        Parameters:
            batch (List[Dict[str, Any]]): A list of data samples where each sample
                                          is a dictionary mapping from string keys to data values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping from string keys to torch.Tensors,
                                     collated from the input batch.
        """
        raise NotImplementedError('Subclasses should implement this method')


@dataclass
class SmartDataCollatorWithPaddinginBatch(AbstractDataCollator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_args: dict,
        model_args: dict,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        mlm: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mlm = mlm
        self.mlm_probability = data_args.mlm_probability

    def __call__(
        self,
        features: list[dict[str, list[int] | list[list[int]] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        special_keys = [
            'input_ids',
            'attention_mask',
            'token_type_ids',
            'mlm_input_ids',
            'mlm_labels',
        ]
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append(
                    {
                        k: feature[k][i] if k in special_keys else feature[k]
                        for k in feature
                    },
                )

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            # padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
            # truncation = True
        )
        if self.model_args.do_mlm:
            batch['mlm_input_ids'], batch['mlm_labels'] = self.mask_tokens(
                batch['input_ids'],
            )

        batch = {
            k: batch[k].view(bs, num_sent, -1)
            if k in special_keys
            else batch[k].view(bs, num_sent, -1)[:, 0]
            for k in batch
        }

        if 'label' in batch:
            batch['labels'] = batch['label']
            del batch['label']
        if 'label_ids' in batch:
            batch['labels'] = batch['label_ids']
            del batch['label_ids']

        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True,
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool,
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(
                labels.shape, 0.8,
            ),
        ).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token,
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
