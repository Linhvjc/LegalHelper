from __future__ import annotations

from transformers import PreTrainedTokenizer

from cse.utils.normalize import normalize_encode
from cse.utils.normalize import normalize_word_diacritic
from cse.utils.normalize import remove_punctuation


def convert_text_to_features(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int | None = 128,
    special_tokens_count: int | None = 2,
    lower_case: bool | None = False,
    remove_punc: bool | None = False,
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
