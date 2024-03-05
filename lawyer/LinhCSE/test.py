from __future__ import annotations

import math

from transformers import AutoTokenizer


def auto_chunker(tokenizer, document, max_chunk_size):

    document_tokens = tokenizer.tokenize(document)
    document_size = len(document_tokens)
    # total chunk number
    K = math.ceil(document_size / max_chunk_size)
    # average integer chunk size
    average_chunk_size = math.ceil(document_size / K)
    # number of chunks with average_chunk_size - 1
    shorter_chunk_number = K * average_chunk_size - document_size
    # number of chunks with average_chunk_size
    standard_chunk_number = K - shorter_chunk_number

    chunks = []
    chunk_start = 0
    for i in range(0, K):
        if i < standard_chunk_number:
            chunk_end = chunk_start + average_chunk_size
        else:
            chunk_end = chunk_start + average_chunk_size - 1
        chunk = document_tokens[chunk_start:chunk_end]
        chunks.append(chunk)
        chunk_start = chunk_end

    assert chunk_start == document_size
    return chunks


tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
document_text = 'Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size Firstly, we decide a maximum_chunk_size maximum_chunk_size based on the LLM context window and the prompt size'
auto_chunker(tokenizer, document_text, 7)
