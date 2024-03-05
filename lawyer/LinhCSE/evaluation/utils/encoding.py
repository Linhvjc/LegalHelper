from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from evaluation.utils.featuring import convert_text_to_features_long_context


def encode(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    pooler,
    texts: list[str],
    max_seq_len: int,
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

    features = tokenizer(
        texts,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        pad_to_multiple_of=max_seq_len,
    )

    with torch.no_grad():
        inputs = {
            'input_ids': features['input_ids'].to(model.device),
            'attention_mask': features['attention_mask'].to(model.device),
            'token_type_ids': features['token_type_ids'].to(model.device),
        }
        outputs = model(**inputs)
        embedding = pooler(inputs['attention_mask'], outputs)

    if return_output == 'np':
        embedding.detach().cpu().numpy()
    return embedding


def encode_long_context(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    pooler,
    texts: list[str],
    max_seq_len: int,
    return_output: str | None = 'np',
    **kwargs,
) -> tuple(np.array(), list):
    """
    Encode a list of texts with long context using a tokenizer and a pre-trained model.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
        model (PreTrainedModel): The pre-trained model to use for encoding the text.
        pooler: The pooling function to apply to the model's output.
        texts (list[str]): A list of texts to encode.
        max_seq_len (int): The maximum sequence length for tokenization.
        return_output (str, optional): The desired output format, either 'np'
        for NumPy array or 'torch' for PyTorch tensor. Defaults to 'np'.
        **kwargs: Additional keyword arguments to be passed to the tokenizer.

    Returns:
        tuple(np.array(), list): A tuple containing the encoded text as a NumPy array and a list of chunk sizes.

    """
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
            lower_case=True,
            remove_punc=True,
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
        outputs = model(**inputs)
        embedding = pooler(batch_attention_mask, outputs)

    if return_output == 'np':
        embedding.detach().cpu().numpy()

    return embedding, count_chunk
