from __future__ import annotations

import numpy as np
import torch
from torch.nn import Module
from transformers import PreTrainedTokenizer

from .base import ModelEncoder
from cse.dataset.featuring import convert_text_to_features


class CSEEncoder(ModelEncoder):

    use_lower_case = True
    use_remove_punc = True

    def __init__(self, pooler_type: str = 'avg') -> None:
        super().__init__(pooler_type)

    def encode(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        texts: list[str], max_seq_len: int,
        return_output: str | None = 'np',
        **kwargs,
    ) -> np.array:
        """
        Encode a list of texts using a tokenizer and a pre-trained model.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
            model (PreTrainedModel): The pre-trained model to use for encoding the text.
            pooler: The pooling function to apply to the model's output.
            texts (list[Text]): A list of texts to encode.
            max_seq_len (int): The maximum sequence length for tokenization.
            return_output (str, optional): The desired output format, either 'np'
            for NumPy array or 'torch' for PyTorch tensor. Defaults to 'np'.
            **kwargs: Additional keyword arguments to be passed to the tokenizer.

        Returns:
            np.array: The encoded text as a NumPy array.

        """

        batch_input_ids, batch_attention_mask = [], []
        for text in texts:
            # Process for single data-point
            (
                input_ids,
                attention_mask,
            ) = convert_text_to_features(
                text=text,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                lower_case=self.use_lower_case,
                remove_punc=self.use_remove_punc,
            )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(
            model.device,
        )
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(
            model.device,
        )
        with torch.no_grad():
            inputs = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
            }
            embedding = model(**inputs)

        if return_output == 'np':
            embedding.detach().cpu().numpy()

        return embedding
