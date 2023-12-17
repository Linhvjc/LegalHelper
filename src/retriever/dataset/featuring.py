from __future__ import annotations

import math
from typing import Any

from transformers import PreTrainedTokenizer

from src.retriever.utils.normalize import normalize_encode
from src.retriever.utils.normalize import normalize_word_diacritic
from src.retriever.utils.normalize import remove_punctuation


def convert_text_to_features(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 128,
    special_tokens_count: int = 2,
    lower_case: bool = False,
    remove_punc: bool = False,
    **kwargs,
) -> tuple[list]:
    """
    Tokenizes and converts the input text into feature vectors suitable for model input.

    Args:
        text (str): The input text to be converted into features.
        tokenizer (PreTrainedTokenizer): The pre-trained tokenizer object to tokenize the input text.
        max_seq_len (int, optional): Maximum sequence length for the output feature vectors.
            Defaults to 128.
        special_tokens_count (int, optional): Number of special tokens like [CLS] and [SEP] in the tokenizer vocabulary.
            Defaults to 2.
        lower_case (bool, optional): Whether to convert the text to lowercase. Defaults to False.
        remove_punc (bool, optional): Whether to remove punctuation from the text. Defaults to False.
        **kwargs: Additional keyword arguments for future expansion.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists - input_ids and attention_mask.
        - input_ids (List[int]): List of token IDs representing the input text with padding if necessary.
        - attention_mask (List[int]): List of attention mask values (1 for real tokens, 0 for padding tokens).

    Raises:
        AssertionError: If the length of input_ids does not match the specified max_seq_len.

    Example:
        text = "Example input text for tokenization."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_mask = convert_text_to_features(text, tokenizer)
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

    # Truncate data
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[: (max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]

    # Add [CLS] token
    tokens = [cls_token] + tokens

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # TODO use smart padding in here
    # Zero-pad up to the sequence length. This is static method padding
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    assert len(input_ids) == max_seq_len, 'Error with input length {} vs {}'.format(
        len(input_ids), max_seq_len,
    )

    return input_ids, attention_mask


def convert_text_to_features_long_context(
    text: str,
    tokenizer: PreTrainedTokenizer,
    mode: str,
    window_size: int,
    max_seq_len: int,
    special_tokens_count: int = 2,
    lower_case: bool = False,
    remove_punc: bool = False,
    **kwargs,
) -> tuple[list]:
    """
    Tokenizes and converts the input text into feature vectors suitable for model input.

    Args:
        text (str): The input text to be converted into features.
        tokenizer (PreTrainedTokenizer): The pre-trained tokenizer object to tokenize the input text.
        max_seq_len (int, optional): Maximum sequence length for the output feature vectors.
            Defaults to 128.
        special_tokens_count (int, optional): Number of special tokens like [CLS] and [SEP] in the tokenizer vocabulary.
            Defaults to 2.
        lower_case (bool, optional): Whether to convert the text to lowercase. Defaults to False.
        remove_punc (bool, optional): Whether to remove punctuation from the text. Defaults to False.
        **kwargs: Additional keyword arguments for future expansion.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists - input_ids and attention_mask.
        - input_ids (List[int]): List of token IDs representing the input text with padding if necessary.
        - attention_mask (List[int]): List of attention mask values (1 for real tokens, 0 for padding tokens).

    Raises:
        AssertionError: If the length of input_ids does not match the specified max_seq_len.

    Example:
        text = "Example input text for tokenization."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_mask = convert_text_to_features(text, tokenizer)
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

    if mode == 'fix':
        len_truncate = (max_seq_len - special_tokens_count)
        all_chunk_tokens = [
            tokens[i:i + len_truncate]
            for i in range(0, len(tokens), len_truncate)
        ]
    elif mode == 'slide window':
        len_truncate = (max_seq_len - special_tokens_count)
        if len(tokens) > window_size:
            all_chunk_tokens = [
                tokens[i - window_size: i - window_size + len_truncate]
                for i in range(window_size, len(tokens), len_truncate)
            ]
        else:
            all_chunk_tokens = [tokens]
    elif mode == 'smart v1':
        # total chunk number
        document_size = len(tokens)
        # average integer chunk size
        k = math.ceil(document_size / max_seq_len)
        average_chunk_size = math.ceil(document_size / k)

        # Truncate data
        len_truncate = (average_chunk_size - special_tokens_count)
        if len_truncate > 0:
            all_chunk_tokens = [
                tokens[i: i + len_truncate]
                for i in range(0, len(tokens), len_truncate)
            ]
        else:
            all_chunk_tokens = [tokens]
    elif mode == 'smart v2':
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
    else:
        raise NotImplementedError

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


def prepare_features(
    examples,
    sent_0_cname: str,
    sent_1_cname: str,
    sent_2_cname: str,
    tokenizer: PreTrainedTokenizer,
    data_args: dict[str, Any],
) -> dict[str, Any]:
    """
    If no sentence in the batch exceed the max length, then use
    the max sentence length in the batch, otherwise use the
    max sentence length in the argument and truncate those that
    exceed the max length.
    padding = max_length (when pad_to_max_length, for pressure test)
    All sentences are padded/truncated to data_args['max_seq_length.
    """
    total = len(examples[sent_0_cname])

    # Avoid "None" fields
    for idx in range(total):
        examples[sent_0_cname][idx] = examples[sent_0_cname][idx] or ' '
        examples[sent_1_cname][idx] = examples[sent_1_cname][idx] or ' '

    sentences = examples[sent_0_cname] + examples[sent_1_cname]

    # If hard negative exists
    if sent_2_cname is not None:
        for idx in range(total):
            examples[sent_2_cname][idx] = examples[sent_2_cname][idx] or ' '
        sentences += examples[sent_2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding='max_length' if data_args.pad_to_max_length else False,
    )

    features = {}
    if sent_2_cname is not None:
        for key in sent_features:
            features[key] = [
                [
                    sent_features[key][i],
                    sent_features[key][i + total],
                    sent_features[key][i + total * 2],
                ]
                for i in range(total)
            ]
    else:
        for key in sent_features:
            features[key] = [
                [sent_features[key][i], sent_features[key][i + total]] for i in range(total)
            ]
    return features
