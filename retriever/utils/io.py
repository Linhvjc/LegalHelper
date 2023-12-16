from __future__ import annotations

import json

import yaml
from datasets import load_dataset


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


def write_json(data, file_path: str, encoding: str = 'utf-8'):
    """
    Serialize `data` to a JSON-formatted file.

    This function takes a data object and serializes it to JSON, saving the output
    to a specified file with a chosen encoding. The JSON is formatted with an
    indentation of 4 spaces.

    Parameters:
    - data (dict): The data to serialize to JSON format.
    - file_path (str): The path to the file where the JSON will be written. If
      the file does not exist, it will be created.
    - encoding (str, optional): The character encoding for the file. Defaults to 'utf-8'.

    Returns:
    - None
    """
    with open(file_path, 'w', encoding=encoding) as pf:
        json.dump(data, pf, ensure_ascii=False, indent=4)


def load_yaml(file_path: str):
    """
    Load configuration from a YAML file.

    This function reads a YAML file from the specified path and parses it into a
    Python dictionary.

    Parameters:
    - file_path (str): The path to the YAML configuration file to be loaded.

    Returns:
    - config (dict): The configuration parameters contained in the YAML file.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)
    return config
