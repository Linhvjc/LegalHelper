from __future__ import annotations

import json
import os

import numpy as np
import torch
from loguru import logger
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from transformers import AutoModel
from transformers import AutoTokenizer

from ..utils.utils import aggregate_score
from ..utils.utils import encode


class Retriever:
    def __init__(self, model_path, database_path, retrieval_max_length) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

        self.embedding_corpus_full = np.load(
            os.path.join(database_path, 'corpus.npy'),
        )
        self.embedding_article = np.load(
            os.path.join(database_path, 'articles.npy'),
        )
        self.count_chunk_docs = np.load(
            os.path.join(database_path, 'counting.npy'),
        )
        with open(os.path.join(database_path, 'corpus.json'), 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        self.num_article_doc = [len(item['sections']) for item in self.corpus]
        self.retrieval_max_length = retrieval_max_length

    def _search_by_bm25(self, query: str, top_best_document) -> str:
        articles_texts = []
        articles_id = []
        texts_splitted = []
        for doc in top_best_document:
            articles_texts.extend(
                [f"Thông tư {doc['document_id']}. {article['content']}" for article in doc['sections']],
            )
            articles_id.extend([
                article['section_id']
                for article in doc['sections']
            ])

            texts_splitted.extend([
                ViTokenizer.tokenize(article['content']) for article in doc['sections']
            ])
        articles_mapping = {
            key: value for key,
            value in zip(articles_texts, articles_id)
        }

        bm25 = BM25Okapi(texts_splitted)
        query = query.lower().split()

        tops_chunk = bm25.get_top_n(
            query, list(articles_mapping.keys()), n=5,
        )
        tops_idx = [articles_mapping[chunk] for chunk in tops_chunk]
        return tops_chunk, tops_idx

    def _search_by_model(self, top_best_document, top_5_best_doc_idx, embedding_query):
        articles_texts = []
        articles_id = []
        for doc in top_best_document:
            articles_texts.extend(
                [f"Thông tư {doc['document_id']}. {article['content']}" for article in doc['sections']],
            )
            articles_id.extend([
                article['section_id']
                for article in doc['sections']
            ])

        articles_embedding = []
        for idx in top_5_best_doc_idx:
            begin_idx = sum(self.num_article_doc[:idx])
            end_idx = begin_idx + \
                self.num_article_doc[idx]
            # best_document = self.corpus[begin_idx:end_idx]
            embedding = self.embedding_article[begin_idx:end_idx]
            articles_embedding.append(embedding)
        articles_embedding = np.concatenate(articles_embedding)

        scores = (
            embedding_query.cpu().detach().numpy() @
            articles_embedding.T
        ).squeeze(0)

        top5_best_doc = sorted(
            range(len(scores)),
            key=lambda i: scores[i], reverse=True,
        )[:5]

        tops_chunk = [articles_texts[idx] for idx in top5_best_doc]
        tops_idx = [articles_id[idx] for idx in top5_best_doc]
        return tops_chunk, tops_idx

    def retrieval(self, text):
        embedding_query = encode(
            tokenizer=self.tokenizer,
            model=self.model,
            text=text,
        )

        scores_chunk, scores_document = aggregate_score(
            embedding_query=embedding_query,
            embedding_corpus_full=self.embedding_corpus_full,
            count_chunk_docs=self.count_chunk_docs,
        )

        top_5_best_doc_idx = sorted(
            range(len(scores_document)),
            key=lambda i: scores_document[i], reverse=True,
        )[:5]
        top_best_document = [self.corpus[idx] for idx in top_5_best_doc_idx]

        #! Get result by model
        tops_chunk_model, tops_idx_model = self._search_by_model(
            top_best_document=top_best_document,
            top_5_best_doc_idx=top_5_best_doc_idx,
            embedding_query=embedding_query,
        )

        #! Get result by bm25
        tops_chunk_bm25, tops_idx_bm25 = self._search_by_bm25(
            query=text,
            top_best_document=top_best_document,
        )

        final_result = ''
        idx_contain = []
        for i in range(len(tops_chunk_model)):

            tops_chunk_model[i] = '' if tops_idx_model[i] in idx_contain else tops_chunk_model[i]
            tops_chunk_bm25[i] = '' if tops_idx_bm25[i] in idx_contain else tops_chunk_bm25[i]

            result_model_pyvi = ViTokenizer.tokenize(tops_chunk_model[i])
            result_bm25_pyvi = ViTokenizer.tokenize(tops_chunk_bm25[i])
            final_result_pyvi = ViTokenizer.tokenize(final_result)

            len_model_tokens = len(self.tokenizer.tokenize(result_model_pyvi))
            len_bm25_tokens = len(self.tokenizer.tokenize(result_bm25_pyvi))
            len_current_tokens = len(
                self.tokenizer.tokenize(final_result_pyvi),
            )

            if (len_current_tokens + len_model_tokens + len_bm25_tokens) < self.retrieval_max_length:
                if len_model_tokens > 0:
                    final_result += tops_chunk_model[i] + '. '
                if len_bm25_tokens > 0:
                    final_result += tops_chunk_bm25[i] + '. '
            elif (len_current_tokens + len_model_tokens) < self.retrieval_max_length:
                if len_model_tokens > 0:
                    final_result += tops_chunk_model[i] + '. '
            else:
                result_model_tokenize = self.tokenizer.tokenize(
                    result_model_pyvi,
                )
                lack_token_len = self.retrieval_max_length - len_current_tokens
                result_model = self.tokenizer.convert_tokens_to_string(
                    result_model_tokenize[:lack_token_len],
                ).replace('_', ' ')
                final_result += result_model
                break
        return final_result


def main_retrieval():
    model_path = '/home/link/spaces/LinhCSE/models/retriever'
    database_path = '/home/link/spaces/LinhCSE/data/concat'
    retriever = Retriever(
        model_path=model_path,
        database_path=database_path,
    )

    while (True):
        input_text = input('Query: ')
        print('Retrieval: ')
        response = retriever.retrieval(input_text, max_length_output=4096)
        print(response)


if __name__ == '__main__':
    main_retrieval()
