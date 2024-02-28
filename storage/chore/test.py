from __future__ import annotations

import os
import statistics
from glob import glob
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from prettytable import PrettyTable
from pyvi import ViTokenizer
from pyvi import ViUtils
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import LongformerModel
from transformers import PhobertTokenizer


model_path = '/home/link/sample/models/best_t11'
benchmark_path = '/home/link/sample/evaluation/benchmark'

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
model = AutoModel.from_pretrained(model_path)
top_k = [1, 5, 10]

# First, check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Then, move your model to the selected device
model = model.to(device)


def print_recall_table(data: dict[str, float]) -> None:
    """
    Print a table displaying recall metrics in a tabular format.

    Args:
        data (dict): A dictionary containing recall metrics as keys and their scores as values.

    Returns:
        None
    """
    # Create a table with headers
    table = PrettyTable(['Benchmark', 'Recall@1', 'Recall@5', 'Recall@10'])
    # Initialize variables to store values for each key
    current_key = None
    recall_1 = None
    recall_5 = None
    recall_10 = None

    # Iterate through the dictionary and populate the table
    for key, value in data.items():
        if '_1' in key:
            current_key = key.replace('_1', '')
            recall_1 = value
        elif '_5' in key:
            recall_5 = value
        elif '_10' in key:
            recall_10 = value
            # Add a row to the table when all values are collected
            table.add_row([current_key, recall_1, recall_5, recall_10])

    # Print the table
    print(table)


def generate_benchmark_filenames(dir_name):
    # List of tuples containing benchmark and corpus filenames
    benchmark_corpus_filenames = []

    # Convention bm_<corpus_name>_<name_of_the_benchmark>
    benchmark_pattern = 'bm'

    # Convention corpus_<corpus_name>
    corpus_pattern = 'corpus'

    corpus_filenames = glob(f'{dir_name}/{corpus_pattern}_*')

    for corpus_filename in corpus_filenames:
        corpus_name = os.path.basename(corpus_filename).split(
            '/',
        )[-1].split('.')[0].split('_')[1]
        benchmark_filenames = glob(
            f'{dir_name}/{benchmark_pattern}_{corpus_name}_*',
        )
        for benchmark_filename in benchmark_filenames:
            benchmark_corpus_filenames.append(
                (benchmark_filename, corpus_filename),
            )

    return benchmark_corpus_filenames


def load_file(file_path: str) -> dict:
    """
    Load data from a file in supported formats (JSON, JSONL, CSV).

    Parameters:
        file_path (str): Path to the file containing the data.

    Returns:
        dict: Loaded data from the specified file.

    Raises:
        Exception: If the file format is not supported.
    """
    supported_formats = ['.json', '.jsonl', '.csv']
    extension = file_path[file_path.rfind('.'):]

    if extension not in supported_formats:
        raise Exception(f"Currently doesn't support for {extension}")

    data_format = 'json' if extension in ['.json', '.jsonl'] else 'csv'
    data = load_dataset(data_format, data_files=file_path)['train']

    return data


def recall(preds: list[str], targets: list[str]) -> float:
    """
    Calculate the recall score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect recall since there are no targets to predict
        else:
            return 0.0  # No recall since there are no targets to predict

    true_positive = len(set(preds) & set(targets))
    recall_score = round(true_positive / len(targets), 4)

    return recall_score


benchmark_corpus_filenames = generate_benchmark_filenames(benchmark_path)
batch_size = 128

results = {}


def encode(model, tokenizer, texts, max_seq_len, return_output='np', **kwargs):
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
        embedding = embedding.last_hidden_state[:, 0, :]  # CLS token
        # embedding = embedding.last_hidden_state.mean(dim=1)  # AVG token

    if return_output == 'np':
        embedding.detach().cpu().numpy()
    return embedding


model.eval()
for paths in tqdm(benchmark_corpus_filenames):
    (benchmark_path, corpus_path) = paths
    print(benchmark_path, corpus_path)
    name_benchmark = benchmark_path.split('/')[-1].split('.')[0]
    benchmark = load_file(benchmark_path).to_list()
    corpus = load_file(corpus_path).to_list()

    embedding_corpus = None
    ids_corpus = []

    for i in tqdm(range(0, len(corpus), batch_size)):
        documents = []
        for doc in corpus[i: i + batch_size]:
            ids_corpus.append(doc['meta']['id'])
            documents.append(ViTokenizer.tokenize(doc['text']))
            # documents.append(ViTokenizer.tokenize(doc['text']))

        embedding = encode(model, tokenizer, documents, 256)

        if embedding_corpus is None:
            embedding_corpus = embedding.detach().cpu().numpy()
        else:
            embedding_corpus = np.append(
                embedding_corpus,
                embedding.detach().cpu().numpy(),
                axis=0,
            )

    # print(embedding_corpus)
    print(embedding_corpus.shape)

    embedding_query = None
    ground_truths = []

    for i in tqdm(range(0, len(benchmark), batch_size)):
        for query in benchmark[i: i + batch_size]:
            queries = []
            ground_truths.append(query['gt'])
            queries.append(ViTokenizer.tokenize(query['query']))

            embedding = encode(model, tokenizer, queries, 256)
            if embedding_query is None:
                embedding_query = embedding.detach().cpu().numpy()
            else:
                embedding_query = np.append(
                    embedding_query,
                    embedding.detach().cpu().numpy(),
                    axis=0,
                )

    # print(embedding_query)
    print(embedding_query.shape)
    scores = embedding_query @ embedding_corpus.T

    for score, ground_truth in zip(scores, ground_truths):
        for k in top_k:
            if f'recall_{name_benchmark}_{k}' not in results:
                results[f'recall_{name_benchmark}_{k}'] = []

            ind = np.argpartition(score, -k)[-k:]

            pred = list(map(ids_corpus.__getitem__, ind))

            results[f'recall_{name_benchmark}_{k}'].append(
                recall(pred, ground_truth),
            )


for k, v in results.items():
    results[k] = statistics.mean(v)
print_recall_table(results)
