from __future__ import annotations

import torch
from retriever.dataset.featuring import convert_text_to_features
from retriever.dataset.featuring import convert_text_to_features_long_context
from torch.nn import Module
from transformers import PreTrainedTokenizer

from .base import ModelEncoder


class KidEncoder(ModelEncoder):

    use_lower_case = True
    use_remove_punc = True

    def __init__(self, pooler_type: str = 'avg') -> None:
        super().__init__(pooler_type)

    def encode(self, model: Module, tokenizer: PreTrainedTokenizer, texts: list[str], max_seq_len: int, return_output: str = 'np', **kwargs):

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

    def encode_long_context(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        mode: str,
        window_size: int,
        texts: list[str],
        max_seq_len: int,
        return_output: str = 'np', **kwargs,
    ):

        count_chunk = []
        batch_input_ids, batch_attention_mask = [], []
        for text in texts:
            # Process for single data-point
            (
                input_ids,
                attention_mask,
                chunk_size,
            ) = convert_text_to_features_long_context(
                text=text,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                lower_case=self.use_lower_case,
                remove_punc=self.use_remove_punc,
                mode=mode,
                window_size=window_size,
            )

            batch_input_ids.extend(input_ids)
            batch_attention_mask.extend(attention_mask)
            count_chunk.append(chunk_size)

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

        return embedding, count_chunk
