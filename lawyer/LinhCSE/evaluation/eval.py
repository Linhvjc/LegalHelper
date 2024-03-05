from __future__ import annotations

import json
from statistics import mean

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

from evaluation.utils.encoding import encode
from evaluation.utils.encoding import encode_long_context
from evaluation.utils.pooling import Pooler
# from evaluation.utils import preprocess_text


class Evaluation():
    def __init__(
            self,
            benchmark_path: str,
            corpus_path: str,
            model_path: str,
            pooler_type: str,
            max_seq_len: int | None = 64,
            max_doc_len: int | None = 256,
            top_k: list | None = [1, 5, 10, 20],
    ) -> None:
        self.benchmark, self.corpus = self._load_data(
            benchmark_path, corpus_path,
        )
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_doc_len = max_doc_len
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.pooler = Pooler(pooler_type=pooler_type)

    def _load_data(self, benchmark_path: str, corpus_path: str):
        with open(benchmark_path) as f:
            benchmark = json.load(f)

        with open(corpus_path) as f:
            corpus = json.load(f)
        return benchmark, corpus

    def _aggregate_score(self, naive_scores: np.array | list, count_chunk: list):
        scores = []
        for score in tqdm(naive_scores):
            score_each_query = []
            cursor = 0
            for i, size in enumerate(count_chunk):
                item = score[cursor:cursor + size]
                item = sorted(item, reverse=True)
                cursor += size
                max_score = item[:min(2, len(item))]
                max_score = max_score[::-1]
                real_score = sum([
                    score * (j + 1 * 2) for j, score in enumerate(
                        max_score,
                    )
                ]) / sum([(k + 1 * 2) for k in range(len(max_score))])
                score_each_query.append(real_score)
            scores.append(score_each_query)
        return scores

    def dynamic_eval(self, batch_size: int | None = 128, top_k: list | None = None, chunking: bool | None = False):
        """
        Perform evaluation on the benchmark and corpus data (Basic).

        Args:
            batch_size (int, optional): The batch size for embedding computation. Defaults to 128.
            top_k (list, optional): A list containing the values of k for top-k evaluation. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation results,
            where the keys are the recall metrics at different k values.
        """
        top_k = top_k or self.top_k
        results = {}
        self.model.eval()

        # --------- Embedding corpus ---------
        embedding_corpus = None
        ids_corpus = []
        count_chunk = []

        logger.info('Embedding corpus')
        for i in tqdm(range(0, len(self.corpus), batch_size)):
            documents = []
            for doc in self.corpus[i: i + batch_size]:
                ids_corpus.append(doc['meta']['id'])
                documents.append(doc['text'])
            if chunking:
                embedding, chunk_size = encode_long_context(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    pooler=self.pooler,
                    texts=documents,
                    max_seq_len=self.max_doc_len,
                )
                count_chunk.extend(chunk_size)
            else:
                embedding = encode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    pooler=self.pooler,
                    texts=documents,
                    max_seq_len=self.max_doc_len,
                )

            if embedding_corpus is None:
                embedding_corpus = embedding.detach().cpu().numpy()
            else:
                embedding_corpus = np.append(
                    embedding_corpus,
                    embedding.detach().cpu().numpy(),
                    axis=0,
                )

        # --------- Embedding queries ---------
        embedding_query = None
        ground_truths = []

        logger.info('Embedding query')
        for i in tqdm(range(0, len(self.benchmark), batch_size)):
            queries = []
            for query in self.benchmark[i: i + batch_size]:
                ground_truths.append(query['gt'])
                queries.append(query['query'])

            embedding = encode(
                model=self.model,
                tokenizer=self.tokenizer,
                pooler=self.pooler,
                texts=queries,
                max_seq_len=self.max_seq_len,
            )

            if embedding_query is None:
                embedding_query = embedding.detach().cpu().numpy()
            else:
                embedding_query = np.append(
                    embedding_query,
                    embedding.detach().cpu().numpy(),
                    axis=0,
                )

        # --------- Normalize embedding ---------
        embedding_query_norm = embedding_query / \
            np.linalg.norm(embedding_query, axis=1, keepdims=True)
        embedding_corpus_norm = embedding_corpus / \
            np.linalg.norm(embedding_corpus, axis=1, keepdims=True)

        # --------- Compute the cosine similarity ---------
        naive_scores = np.dot(
            embedding_query_norm,
            embedding_corpus_norm.T,
        )
        if chunking:
            logger.info('Aggregate scores')
            scores = self._aggregate_score(naive_scores, count_chunk)
        else:
            scores = naive_scores

        # --------- Compute recall ---------
        for score, ground_truth in zip(scores, ground_truths):
            for k in self.top_k:
                if f'recall_{k}' not in results:
                    results[f'recall_{k}'] = []

                ind = np.argpartition(score, -k)[-k:]
                pred = list(map(ids_corpus.__getitem__, ind))

                true_positive = len(set(pred) & set(ground_truth))

                if k == 1:
                    recall = 1 if true_positive == 1 else 0
                else:
                    recall = true_positive / len(ground_truth)
                results[f'recall_{k}'].append(recall)
        for k, v in results.items():
            results[k] = mean(v)
        return results

    def tf_idf(self):
        results = {}
        all_corpus = [doc['text'] for doc in self.corpus]
        all_queris = [item['query'] for item in self.benchmark]
        ground_truths = [item['gt'] for item in self.benchmark]
        ids_corpus = [doc['meta']['id'] for doc in self.corpus]

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(all_corpus)

        tfidf_corpus = vectorizer.transform(all_corpus)
        tfidf_queries = vectorizer.transform(all_queris)

        scores = np.dot(
            tfidf_queries,
            tfidf_corpus.T,
        )
        scores = scores.toarray()

        for score, ground_truth in zip(scores, ground_truths):
            for k in self.top_k:
                if f'recall_{k}' not in results:
                    results[f'recall_{k}'] = []

                ind = np.argpartition(score, -k)[-k:]
                pred = list(map(ids_corpus.__getitem__, ind))

                true_positive = len(set(pred) & set(ground_truth))

                if k == 1:
                    recall = 1 if true_positive == 1 else 0
                else:
                    recall = true_positive / len(ground_truth)
                results[f'recall_{k}'].append(recall)
        for k, v in results.items():
            results[k] = mean(v)
        return results


if __name__ == '__main__':
    benchmark_path = '/home/link/spaces/chunking/LinhCSE_training/benchmark/zalo/bm_legal_zalo.json'
    corpus_path = '/home/link/spaces/chunking/LinhCSE_training/benchmark/zalo/corpus_legal.json'
    model_path = 'linhphanff/phobert-cse-general'
    evaluation = Evaluation(
        benchmark_path=benchmark_path,
        corpus_path=corpus_path,
        model_path=model_path,
        pooler_type='avg',
    )

    # results = evaluation.dynamic_eval(batch_size=64, chunking=True)
    results = evaluation.tf_idf()
    print(results)
