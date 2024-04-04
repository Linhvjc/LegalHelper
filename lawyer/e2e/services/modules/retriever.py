from __future__ import annotations

import json
import os

import numpy as np
import torch
from pyvi import ViTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
from typing import Optional

from ..utils.utils import aggregate_score
from ..utils.utils import encode
from ..utils.bm25 import BM25Plus


class Retriever:
    def __init__(self,
                 model_path: Optional[str] = None,
                 database_path: Optional[str] = None,
                 retrieval_max_length: Optional[int] = None
                 ) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        # self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        if database_path:
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
            self.num_article_doc = [len(item['sections'])
                                    for item in self.corpus]
            self.retrieval_max_length = retrieval_max_length

    async def _search_by_bm25(self, query: str, top_best_document) -> str:
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

        bm25 = BM25Plus(texts_splitted)
        query = query.lower().split()

        tops_scores = bm25.get_scores(query)
        tops_idx = np.argsort(-tops_scores)[:5]
        tops_chunk = [list(articles_mapping.keys())[idx] for idx in tops_idx]
        return tops_chunk, tops_idx

    async def _search_by_model(self, top_best_document, top_5_best_doc_idx, embedding_query):
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

    async def retrieval(self, text):
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
        tops_chunk_model, tops_idx_model = await self._search_by_model(
            top_best_document=top_best_document,
            top_5_best_doc_idx=top_5_best_doc_idx,
            embedding_query=embedding_query,
        )

        #! Get result by bm25
        tops_chunk_bm25, tops_idx_bm25 = await self._search_by_bm25(
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

    def retrieval_tool(self, docs: list, query: str, output_length):
        embedding_query = encode(
            tokenizer=self.tokenizer,
            model=self.model,
            text=query,
        )
        embedding_corpus = None
        for i in range(0, len(docs), 16):
            embedding = self._encode(
                texts=docs[i:i+16],
                max_seq_len=256
            )

            if embedding_corpus is None:
                embedding_corpus = embedding.detach().cpu().numpy()
            else:
                embedding_corpus = np.append(
                    embedding_corpus,
                    embedding.detach().cpu().numpy(),
                    axis=0,
                )
        scores = (
            embedding_query.cpu().detach().numpy() @
            embedding_corpus.T
        ).squeeze(0)

        top_best_doc = sorted(
            range(len(scores)),
            key=lambda i: scores[i], reverse=True,
        )
        docs_relevant_sort = [docs[idx] for idx in top_best_doc]
        final_result = ""
        for item in docs_relevant_sort:
            if len(final_result.split()) + len(item.split()) < output_length:
                final_result += f"{item}. "

        return final_result

    def _encode(self, texts, max_seq_len):
        texts = [ViTokenizer.tokenize(text) for text in texts]
        features = self.tokenizer(
            texts,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            pad_to_multiple_of=max_seq_len
        )
        # with torch.no_grad():
        inputs = {
            'input_ids': features['input_ids'].to(self.model.device),
            'attention_mask': features['attention_mask'].to(self.model.device),
            'token_type_ids': features['token_type_ids'].to(self.model.device),
        }
        embedding_query = self.model(**inputs)
        embedding_query = embedding_query.last_hidden_state.mean(
            dim=1,
        )  # AVG token

        return embedding_query


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
    print('abc')
