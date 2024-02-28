from __future__ import annotations

import json

import numpy as np
import torch
from pyvi import ViTokenizer
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

from src.retriever.utils.normalize import normalize_encode
from src.retriever.utils.normalize import normalize_word_diacritic
from src.retriever.utils.normalize import remove_punctuation

model_path = '/home/link/spaces/LinhCSE/models/retriever'
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
model = AutoModel.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def pre_process(text):
    text = normalize_encode(normalize_word_diacritic(text))
    text = text.lower()
    text = remove_punctuation(text)
    return text


def encode(texts, max_seq_len=256, return_output='np', **kwargs):
    features = tokenizer(
        texts,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        pad_to_multiple_of=256,
    )
    with torch.no_grad():
        inputs = {
            'input_ids': features['input_ids'].to(model.device),
            'attention_mask': features['attention_mask'].to(model.device),
            'token_type_ids': features['token_type_ids'].to(model.device),
        }
        embedding = model(**inputs)
        # embedding = embedding.last_hidden_state[:, 0, :]  # CLS token
        embedding = embedding.last_hidden_state.mean(dim=1)  # AVG token

    if return_output == 'np':
        embedding.detach().cpu().numpy()
    return embedding


with open('/home/link/spaces/LinhCSE/data/corpus.json') as f:
    corpus = json.load(f)

all_texts = []
for document in tqdm(corpus):
    for chunk in document['section']:
        text = ViTokenizer.tokenize(chunk['content'])
        text = pre_process(text)
        all_texts.append(text)

embedding_query = None
for i in tqdm(range(0, len(all_texts), 512)):
    # for query in all_texts[i: i + 64]:
    queries = all_texts[i: i + 512]
    # queries.append(query)

    embedding = encode(queries)
    if embedding_query is None:
        embedding_query = embedding.detach().cpu().numpy()
    else:
        embedding_query = np.append(
            embedding_query,
            embedding.detach().cpu().numpy(),
            axis=0,
        )

with torch.no_grad():
    for document in tqdm(corpus):
        for chunk in document['section']:
            text = ViTokenizer.tokenize(chunk['content'])
            text = pre_process(text)

            embedding_str = np.array2string(embedding_query[0], separator=', ')
            chunk['embedding'] = embedding_str
            embedding_query = embedding_query[1:]

with open('/home/link/spaces/LinhCSE/data/corpus_embedding.json', 'w', encoding='utf-8') as file:
    json.dump(corpus, file, ensure_ascii=False)
