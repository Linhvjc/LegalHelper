from __future__ import annotations

import json

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer


class Retriever:
    def __init__(self, model_path, corpus_path, embedding_path) -> None:
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
        self.model = AutoModel.from_pretrained(model_path)
        self.embedding_corpus = np.load(embedding_path)
        self.count_chunk, self.items_corpus = self._read_corpus(corpus_path)
        self.model = self.model.to(self.device)

    def _read_corpus(self, corpus_path):
        logger.info('Load corpus...')
        with open(corpus_path) as f:
            corpus = json.load(f)

        count_chunk, items_corpus = [], []
        for document in corpus:
            count_chunk.append(len(document['sections']))
            items_corpus.extend(document['sections'])

        return count_chunk, items_corpus

    def retrieval(self, text):
        features = self.tokenizer(
            text=text,
            max_length=64,
            truncation=True,
            return_tensors='pt',
        )
        # with torch.no_grad():
        inputs = {
            'input_ids': features['input_ids'].to(self.device),
            'attention_mask': features['attention_mask'].to(self.device),
            'token_type_ids': features['token_type_ids'].to(self.device),
        }
        embedding_query = self.model(**inputs)
        embedding_query = embedding_query.last_hidden_state.mean(
            dim=1,
        )  # AVG token

        scores = (
            embedding_query.detach().numpy() @
            self.embedding_corpus.T
        ).squeeze(0)
        logger.info('Normalize scores')
        scores_normalize = []
        cursor = 0
        for size in self.count_chunk:
            item = scores[cursor:cursor + size]
            item = sorted(item, reverse=True)
            cursor += size
            max_score = item[:min(2, len(item))]
            max_score = max_score[::-1]
            real_score = sum(
                [score * (j + 1 * 2) for j, score in enumerate(max_score)],
            ) / sum([(k + 1 * 2) for k in range(len(max_score))])
            scores_normalize.append(real_score)

        index_best_doc, _ = max(
            enumerate(scores_normalize), key=lambda x: x[1],
        )
        position_begin_best_doc = sum(self.count_chunk[:index_best_doc])
        position_end_best_doc = position_begin_best_doc + \
            self.count_chunk[index_best_doc]

        # best_chunks_score = scores[position_begin_best_doc:position_end_best_doc]
        # index_best_chunk, _ = max(enumerate(best_chunks_score), key=lambda x: x[1])
        # result = self.items_corpus[position_begin_best_doc + index_best_chunk]

        best_document = self.items_corpus[position_begin_best_doc:position_end_best_doc]
        result = ''
        for chunk in best_document:
            result += f"{chunk['content']}. "

        return result


if __name__ == '__main__':
    model_path = '/home/link/spaces/LinhCSE/models/retriever'
    corpus_path = '/home/link/spaces/LinhCSE/data/full/corpus.json'
    embedding_path = '/home/link/spaces/LinhCSE/data/full/embeddings_corpus.npy'
    retriever = Retriever(
        model_path=model_path, corpus_path=corpus_path, embedding_path=embedding_path,
    )

    while (True):
        input_text = input('Query: ')
        print('Retrieval: ')
        response = retriever.retrieval(input_text)
        print(response)
