from __future__ import annotations

import argparse
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import DatasetDict
from datasets import load_dataset
from prettytable import PrettyTable
from pyvi import ViTokenizer
from tabulate import tabulate
from transformers import AutoTokenizer
from transformers import RobertaConfig
from transformers import set_seed

import wandb
from cse.models.dense_model import DenseRoberta


MODEL_CLASSES = {
    'kid-dense-phobert-base': (RobertaConfig, DenseRoberta, AutoTokenizer),
    'kid-dense-phobert-base-v2': (RobertaConfig, DenseRoberta, AutoTokenizer),
    'kid-dense-phobert-large': (RobertaConfig, DenseRoberta, AutoTokenizer),
    'kid-dense-sim-cse-vietnamese': (RobertaConfig, DenseRoberta, AutoTokenizer),
    'kid-dense-unsim-cse-vietnamese': (RobertaConfig, DenseRoberta, AutoTokenizer),
}

MODEL_PATH_MAP = {
    'kid-dense-phobert-base': 'vinai/phobert-base',
    'kid-dense-phobert-base-v2': 'vinai/phobert-base-v2',
    'kid-dense-phobert-large': 'vinai/phobert-large',
    'kid-dense-sim-cse-vietnamese': 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
    'kid-dense-unsim-cse-vietnamese': 'VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base',
}


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path, use_fast=True,
    )


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


vowel = [
    ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
    ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
    ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
    ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
    ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
    ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
    ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
    ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
    ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
    ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
    ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
    ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y'],
]

vowel_to_idx = {}
for i in range(len(vowel)):
    for j in range(len(vowel[i]) - 1):
        vowel_to_idx[vowel[i][j]] = (i, j)


def print_recall_table(data: dict[str, float]) -> None:
    """
    Print a table displaying recall metrics in a tabular format.

    Args:
        data (dict): A dictionary containing recall metrics as keys and their scores as values.

    Returns:
        None
    """
    # Create a table with headers
    table = PrettyTable(['Benchmark', 'Recall@5', 'Recall@10', 'Recall@20'])
    # Initialize variables to store values for each key
    current_key = None
    recall_5 = None
    recall_10 = None
    recall_20 = None

    # Iterate through the dictionary and populate the table
    for key, value in data.items():
        if '_5' in key:
            current_key = key.replace('_5', '')
            recall_5 = value
        elif '_10' in key:
            recall_10 = value
        elif '_20' in key:
            recall_20 = value
            # Add a row to the table when all values are collected
            table.add_row([current_key, recall_5, recall_10, recall_20])

    # Print the table
    print(table)


def print_parse_arguments(args: argparse.Namespace, output_format: str = 'line') -> None | str:
    """
    Parse the command line arguments using the provided argparse.Namespace object.

    Args:
        args (argparse.Namespace): An argparse.Namespace object containing parsed command line arguments.
        output_format (str): Output format for parsed arguments. Options: 'line' or 'table'. Default is 'line'.

    Returns:
        Union[None, str]: None if invalid output format specified, else a string representing parsed arguments.
    """
    # Convert parsed arguments to dictionary
    parsed_args = vars(args)
    parsed_args = dict(sorted(parsed_args.items(), key=lambda x: x[0]))

    # Output parsed arguments based on the specified format
    if output_format == 'line':
        # Output arguments line by line
        output = '\n'.join(
            [f'{key}: {value}' for key, value in parsed_args.items()],
        )
    elif output_format == 'table':
        # Output arguments in table format
        output = tabulate(parsed_args.items(), headers=['Argument', 'Value'])
    else:
        raise Exception(
            "Invalid output format specified. Please choose 'line' or 'table'.",
        )

    print(output)


def log_embeddings_to_wandb(embeddings: np.array = None, name: str = ''):
    """
    Log embeddings to Weights & Biases (wandb) platform.

    Parameters:
        embeddings (np.array): The embeddings to be logged, should be a 2D numpy array.
        name (Text): A string specifying the name of the embeddings being logged.

    Returns:
        None
    """
    # Generate column names for the embeddings
    columns_name = [f'D{str(i)}' for i in range(len(embeddings[0]))]

    # Log embeddings to wandb
    wandb.log({
        f'query_embeddings_{name}': wandb.Table(columns=columns_name, data=embeddings.tolist()),
    })


def log_scores_similarity_to_wandb(scores: np.array = None):
    """
    Log similarity scores as a heatmap to Weights & Biases (wandb) dashboard.

    Parameters:
        scores (np.array): 2D array representing similarity scores.
    """

    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Similarity Scores Heatmap')
    plt.xlabel('Queries')
    plt.ylabel('Documents')

    # Log the heatmap as an image to wandb
    wandb.log({'Scores/Similarity Scores Heatmap': plt})

    # Close the plot to prevent displaying it in your console (optional)
    plt.close()


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_valid_vietnam_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, _ = vowel_to_idx.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True
