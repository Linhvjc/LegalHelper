from typing import Text, List
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import json

import pandas as pd
from pyvi import ViTokenizer
from datasets import load_dataset

from utils import logger


def read_csv(path: Text = None) -> pd.DataFrame:
    if (not path) or (Path(path).suffix not in [".csv"]):
        raise ValueError("Please check Path is not None or Path is ended with (.csv)")

    data = pd.read_csv(path)
    return data

def read_asym_data(file_path: Text):
    data = load_dataset('json', data_files=file_path, num_proc=16)['train']
    # result = []
    # logger.info("Load data to list")
    # for i in tqdm(range(data.num_rows)):
    #     result.append(data[i])
    return data

def read_json(path: Text = None):
    if (not path) or (Path(path).suffix not in [".json"]):
        raise ValueError("Please check Path is not None or Path is ended with (.json)")

    json_data = None
    with open(path, "r") as file:
        json_data = json.load(file)
    return json_data


def save_to_json(data, file_path: Text, indent: int = 2):
    if (not file_path) or (Path(file_path).suffix not in [".json"]):
        raise ValueError("Please check Path is not None or Path is ended with (.json)")

    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=indent, ensure_ascii=False)

    logger.info(f"Save data successfully to path: {file_path}")


def process_line(line):
    return ViTokenizer.tokenize(line)


def read_txt(file_path: Text, min_length: int = 10, apply_vi_tokenizer: bool = False):
    lines = []
    with open(file_path, 'r', encoding='utf8', errors='ignore') as fIn:
        for line in tqdm(fIn):
            line = line.strip()
            if len(line) >= min_length:
                lines.append(line)


    if apply_vi_tokenizer:
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes - 3) as pool:
            sentences = pool.map(process_line, lines)
            return sentences

    return lines


def write_to_txt(file_path: Text, data: List[Text]) -> None:
    with open(file_path, 'w') as file:
        for text in data:
            file.write(text + "\n")
    logger.info(f"Write data to {file_path} successfully!!")


def json_dumps(obj, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)
