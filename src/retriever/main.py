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


class Retriever:
    def __init__(self, model_path, database_path) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)

        self.embedding_corpus_full = np.load(
            os.path.join(database_path, 'corpus.npy'),
        )
        self.embedding_article = np.load(
            os.path.join(database_path, 'articles.npy'),
        )
        self.count_chunk_docs = np.load(
            os.path.join(database_path, 'counting.npy'),
        )
        with open(os.path.join(database_path, 'corpus.json')) as f:
            self.corpus = json.load(f)
        self.num_article_doc = [len(item['sections']) for item in self.corpus]

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
        # bm25 = BM25Okapi(corpus_splitted)
        # query = query.lower().split()

        # corpus_splitted, corpus_original = [], {}
        # for i, item in enumerate(items):
        #     text = item['title'] + '. ' + item['content']
        #     corpus_original[text] = i
        #     text = ViTokenizer.tokenize(text)
        #     corpus_splitted.append(text.lower().split())
        # bm25 = BM25Okapi(corpus_splitted)

        # query = query.lower().split()

        # top_result = bm25.get_top_n(
        #     query, list(corpus_original.keys()), n=1,
        # )[0]
        # top_result_index = corpus_original[top_result]

        # return top_result_index, top_result

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

    def _caculate_score(self, embedding_query):
        scores_chunk = (
            embedding_query.cpu().detach().numpy() @
            self.embedding_corpus_full.T
        ).squeeze(0)

        scores_document = []
        cursor = 0
        for size in self.count_chunk_docs:
            item = scores_chunk[cursor:cursor + size]
            item = sorted(item, reverse=True)
            cursor += size
            max_score = item[:min(2, len(item))]
            max_score = max_score[::-1]
            real_score = sum(
                [score * (j + 1 * 2) for j, score in enumerate(max_score)],
            ) / sum([(k + 1 * 2) for k in range(len(max_score))])
            scores_document.append(real_score)

        return scores_chunk, scores_document

    def retrieval(self, text):
        text = ViTokenizer.tokenize(text)
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

        scores_chunk, scores_document = self._caculate_score(
            embedding_query=embedding_query,
        )

        # index_best_doc, _ = max(
        #     enumerate(scores_document), key=lambda x: x[1],
        # )
        top_5_best_doc_idx = sorted(
            range(len(scores_document)),
            key=lambda i: scores_document[i], reverse=True,
        )[:5]
        top_best_document = [self.corpus[idx] for idx in top_5_best_doc_idx]

        #! Get all chunk in best document
        # position_begin_best_doc = sum(self.num_article_doc[:index_best_doc])
        # position_end_best_doc = position_begin_best_doc + \
        #     self.num_article_doc[index_best_doc]
        # best_document = self.corpus[position_begin_best_doc:position_end_best_doc]
        # best_articles_embedding = self.embedding_article[position_begin_best_doc:position_end_best_doc]

        #! Get the title of the document
        # result_title = f"Thông tư số {self.corpus[index_best_doc]['document_id']}"

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

        print('Model: \n', tops_idx_model)
        print('Bm25: \n', tops_idx_bm25)

        # if id_result_model != id_result_bm25:
        #     final_result = f"""{result_title}\nDocument 1: {result_model_top1}\nDocument 2: {result_bm25}"""
        # else:
        #     final_result = f"""{result_title}\nDocument 1: {result_model_top2}\nDocument 2: {result_bm25}"""

        # return final_result


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
        response = retriever.retrieval(input_text)
        print(response)


if __name__ == '__main__':
    main_retrieval()
