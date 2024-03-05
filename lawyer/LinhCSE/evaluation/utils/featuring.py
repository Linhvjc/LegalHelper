from __future__ import annotations

import math

from transformers import PreTrainedTokenizer

from evaluation.utils.normalize import normalize_encode
from evaluation.utils.normalize import normalize_word_diacritic
from evaluation.utils.normalize import remove_punctuation


def convert_text_to_features_long_context(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    special_tokens_count: int | None = 2,
    lower_case: bool | None = False,
    remove_punc: bool | None = False,
    **kwargs,
):
    """
    Convert a given text into input features for long-context tasks.

    Args:
        text (str): The input text to convert.
        tokenizer (PreTrainedTokenizer): The tokenizer object used to tokenize the text.
        max_seq_len (int): The maximum sequence length allowed for each chunk.
        special_tokens_count (int | None, optional): The number of special tokens
        (e.g., [CLS], [SEP]) to consider. Defaults to 2.
        lower_case (bool | None, optional): Whether to convert the text to lowercase. Defaults to False.
        remove_punc (bool | None, optional): Whether to remove punctuation from the text. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the tokenizer.

    Returns:
        Tuple[List[List[int]], List[List[int]], int]: A tuple containing:
            - all_chunk_input_ids (List[List[int]]): A list of input token IDs for each chunk.
            - all_chunk_attention_mask (List[List[int]]): A list of attention masks for each chunk.
            - num_chunks (int): The total number of chunks.

    """
    unk_token = tokenizer.unk_token

    cls_token = tokenizer.cls_token

    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.pad_token_id

    # Normalize text
    text = normalize_encode(normalize_word_diacritic(text))

    if lower_case:
        text = text.lower()
    if remove_punc:
        text = remove_punctuation(text)

    text = text.split()  # Some are spaced twice

    tokens = []
    # Tokenizer
    for word in text:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)

    document_size = len(tokens)
    # total chunk number
    k = math.ceil(document_size / max_seq_len)
    # average integer chunk size
    average_chunk_size = math.ceil(document_size / k)
    # number of chunks with average_chunk_size - 1
    shorter_chunk_number = k * average_chunk_size - document_size
    # number of chunks with average_chunk_size
    standard_chunk_number = k - shorter_chunk_number
    len_truncate = (average_chunk_size - special_tokens_count)

    if len_truncate > 0:
        all_chunk_tokens = []
        chunk_start = 0
        for i in range(0, k):
            if i < standard_chunk_number:
                chunk_end = chunk_start + len_truncate
            else:
                chunk_end = chunk_start + len_truncate - 1
            chunk = tokens[chunk_start:chunk_end]
            all_chunk_tokens.append(chunk)
            chunk_start = chunk_end
    else:
        all_chunk_tokens = [tokens]

    # Add [SEP] token and [CLS] token
    all_chunk_tokens = [
        [cls_token] + chunk + [sep_token]
        for chunk in all_chunk_tokens
    ]

    # Convert tokens to ids
    all_chunk_input_ids = [
        tokenizer.convert_tokens_to_ids(
            chunk,
        ) for chunk in all_chunk_tokens
    ]
    all_chunk_attention_mask = [
        [1] * len(input_ids) for input_ids in all_chunk_input_ids
    ]

    # TODO use smart padding in here
    # Zero-pad up to the sequence length. This is static method padding
    padding_lengths = [
        max_seq_len - len(input_ids)
        for input_ids in all_chunk_input_ids
    ]
    for i in range(len(all_chunk_input_ids)):
        all_chunk_input_ids[i] = all_chunk_input_ids[i] + \
            ([pad_token_id] * padding_lengths[i])
        all_chunk_attention_mask[i] = all_chunk_attention_mask[i] + \
            ([0] * padding_lengths[i])

    return all_chunk_input_ids, all_chunk_attention_mask, len(all_chunk_input_ids)
