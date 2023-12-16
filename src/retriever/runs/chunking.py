from __future__ import annotations

import statistics
from statistics import mean

import numpy as np
import torch
from loguru import logger
from retriever.arguments import KidDatagArguments
from retriever.arguments import KidModelArguments
from retriever.arguments import KidTrainingArguments
from retriever.encode.kid import KidEncoder
from retriever.utils.io import load_file
from retriever.utils.metric import recall
from retriever.utils.utils import generate_benchmark_filenames
from retriever.utils.utils import load_tokenizer
from retriever.utils.utils import MODEL_CLASSES
from retriever.utils.utils import MODEL_PATH_MAP
from retriever.utils.utils import print_recall_table
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Chunking:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        benchmark_dir: str,
        eval_batch_size: int = 128,
        max_seq_len_document: int = 256,
        max_seq_len_query: int = 64,
        top_k_results: list[int] = [5, 10, 20],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmark_corpus_filenames = generate_benchmark_filenames(
            benchmark_dir,
        )
        self.eval_batch_size = eval_batch_size
        self.max_seq_len_document = max_seq_len_document
        self.max_seq_len_query = max_seq_len_query
        self.top_k_results = top_k_results
        self.encoder = KidEncoder()

    def evaluation(self, mode: str, window_size: str, formula: str):
        results = {}
        self.model.eval()

        for paths in self.benchmark_corpus_filenames:
            (benchmark_path, corpus_path) = paths
            name_benchmark = benchmark_path.split('/')[-1].split('.')[0]
            benchmark = load_file(benchmark_path).to_list()
            corpus = load_file(corpus_path).to_list()
            logger.info(f'Benchmark name: {name_benchmark}')

            embedding_corpus = None
            ids_corpus = []
            count_chunk = []

            logger.info('Embedding corpus')
            for i in tqdm(range(0, len(corpus), self.eval_batch_size)):
                documents = []
                for doc in corpus[i: i + self.eval_batch_size]:
                    ids_corpus.append(doc['meta']['id'])
                    documents.append(doc['text'])

                embedding, chunk_size = self.encoder.encode_long_context(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=documents,
                    max_seq_len=self.max_seq_len_document,
                    mode=mode,
                    window_size=window_size,
                )

                if embedding_corpus is None:
                    embedding_corpus = embedding.detach().cpu().numpy()
                else:
                    embedding_corpus = np.append(
                        embedding_corpus,
                        embedding.detach().cpu().numpy(),
                        axis=0,
                    )
                count_chunk.extend(chunk_size)

            embedding_query = None
            ground_truths = []

            logger.info('Embedding query')
            for i in tqdm(range(0, len(benchmark), self.eval_batch_size)):
                queries = []
                for query in benchmark[i: i + self.eval_batch_size]:
                    ground_truths.append(query['gt'])
                    queries.append(query['query'])

                embedding = self.encoder.encode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    texts=queries,
                    max_seq_len=self.max_seq_len_query,
                )

                if embedding_query is None:
                    embedding_query = embedding.detach().cpu().numpy()
                else:
                    embedding_query = np.append(
                        embedding_query,
                        embedding.detach().cpu().numpy(),
                        axis=0,
                    )

            scores_chunks = embedding_query @ embedding_corpus.T

            logger.info('Normalize scores')
            scores = []
            if formula == '2top1_1top2':
                # Calculate score
                for score in tqdm(scores_chunks):
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
            elif formula == 'counting':
                # Count
                for score in tqdm(scores_chunks):
                    score_each_query = []
                    cursor = 0
                    for i, size in enumerate(count_chunk):
                        item = score[cursor:cursor + size]
                        item = sorted(item, reverse=True)
                        cursor += size
                        max_item_length = min(100, len(item))
                        item_cat = item[:max_item_length]
                        count = sum(1 for element in item_cat if element > 0.5)
                        real_score = count / max_item_length * item[0]

                        score_each_query.append(real_score)
                    scores.append(score_each_query)
            elif formula == 'average':
                for score in tqdm(scores_chunks):
                    score_each_query = []
                    cursor = 0
                    for i, size in enumerate(count_chunk):
                        item = score[cursor:cursor + size]
                        mean_score = mean(item)
                        cursor += size

                        score_each_query.append(mean_score)
                    scores.append(score_each_query)
            elif formula == 'max':
                for score in tqdm(scores_chunks):
                    score_each_query = []
                    cursor = 0
                    for i, size in enumerate(count_chunk):
                        item = score[cursor:cursor + size]
                        max_score = max(item)
                        cursor += size

                        score_each_query.append(max_score)
                    scores.append(score_each_query)
            else:
                raise NotImplementedError

            for score, ground_truth in zip(scores, ground_truths):
                for k in self.top_k_results:
                    if f'recall_{name_benchmark}_{k}' not in results:
                        results[f'recall_{name_benchmark}_{k}'] = []

                    ind = np.argpartition(score, -k)[-k:]
                    pred = list(map(ids_corpus.__getitem__, ind))

                    results[f'recall_{name_benchmark}_{k}'].append(
                        recall(pred, ground_truth),
                    )
        return results


if __name__ == '__main__':
    parser = HfArgumentParser(
        (KidDatagArguments, KidModelArguments, KidTrainingArguments),
    )
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()
    train_args.device_map = 'auto'
    train_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_args.model_name_or_path = MODEL_PATH_MAP[model_args.model_type]

    tokenizer = load_tokenizer(model_args)
    config_class, model_class, _ = MODEL_CLASSES[model_args.model_type]

    model = model_class.from_pretrained(
        model_args.pretrained_path,
        torch_dtype=model_args.compute_dtype,
        device_map=train_args.device_map,
        args=model_args,
    )
    # model = AutoModel.from_pretrained(model_args.pretrained_path)

    chunking = Chunking(
        model=model,
        tokenizer=tokenizer,
        benchmark_dir=data_args.benchmark_dir,
        eval_batch_size=64,
        max_seq_len_document=64,
    )

    results = chunking.evaluation(
        mode='smart v2', formula='2top1_1top2', window_size=64,
    )
    for k, v in results.items():
        results[k] = statistics.mean(v)
    print_recall_table(results)
