import multiprocessing
from typing import List

import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from pyvi import ViTokenizer
from sklearn.utils.class_weight import compute_class_weight

from utils.io import read_txt, read_json, read_asym_data
from utils import logger


class OnlineDataset(Dataset):
    def __init__(
            self,
            text_path: str = None,
            student_tokenizer: PreTrainedTokenizer = None,
            teacher_tokenizer: PreTrainedTokenizer = None,
            max_seq_len: int = 128,
            use_vi_tokenizer: bool = False
    ):
        super(OnlineDataset, self).__init__()

        if use_vi_tokenizer:
            logger.info("USE VI TOKENIZER: True")
        self.teacher_text = read_txt(text_path, apply_vi_tokenizer=False)
        self.student_text = read_txt(text_path, apply_vi_tokenizer=use_vi_tokenizer)

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        sample_teacher_text = self.teacher_text[index].strip()
        sample_student_text = self.student_text[index].strip()
        student_inputs = self.student_tokenizer(
            sample_student_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        teacher_inputs = self.teacher_tokenizer(
            sample_teacher_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        for k in student_inputs:
            student_inputs[k] = student_inputs[k].squeeze(0)
        for k in teacher_inputs:
            teacher_inputs[k] = teacher_inputs[k].squeeze(0)

        return student_inputs, teacher_inputs

    def __len__(self):
        return len(self.teacher_text)

class OnlineDatasetAsym(Dataset):
    def __init__(
            self,
            text_path: str = None,
            student_tokenizer: PreTrainedTokenizer = None,
            teacher_tokenizer: PreTrainedTokenizer = None,
            max_query_len: int = 64,
            max_doc_len: int = 128,
            use_vi_tokenizer: bool = False
    ):
        super(OnlineDatasetAsym, self).__init__()

        if use_vi_tokenizer:
            logger.info("USE VI TOKENIZER: True")
        self.texts = read_asym_data(text_path)

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __getitem__(self, index):
        query = self.texts[index]['query']
        document = self.texts[index]['document']

        student_inputs_query = self.student_tokenizer(
            query,
            max_length=self.max_query_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        student_inputs_document = self.student_tokenizer(
            document,
            max_length=self.max_doc_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        
        query = query.replace("_", " ")
        document = document.replace("_", " ")
        teacher_inputs_query = self.teacher_tokenizer(
            query,
            max_length=self.max_query_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        teacher_inputs_document = self.teacher_tokenizer(
            document,
            max_length=self.max_doc_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        for k in student_inputs_query:
            student_inputs_query[k] = student_inputs_query[k].squeeze(0)
        for k in student_inputs_document:
            student_inputs_document[k] = student_inputs_document[k].squeeze(0)
        for k in teacher_inputs_query:
            teacher_inputs_query[k] = teacher_inputs_query[k].squeeze(0)
        for k in teacher_inputs_document:
            teacher_inputs_document[k] = teacher_inputs_document[k].squeeze(0)

        return (
            student_inputs_query,     
            student_inputs_document, 
            teacher_inputs_query, 
            teacher_inputs_document
        )

    def __len__(self):
        return len(self.texts)


def process_single_line(line):
    return ViTokenizer.tokenize(line)


def process_multiple_line(lines):
    line_0 = ViTokenizer.tokenize(lines[0])
    line_1 = ViTokenizer.tokenize(lines[1])
    line_2 = ViTokenizer.tokenize(lines[2])
    return [line_0, line_1, line_2]


class NLIDataset(Dataset):
    """
    File: .json
    Data Structure: sent_1, sent_2, hard_neg
    """

    def __init__(self,
                 data_path: str = None,
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_len: int = 128,
                 use_vi_tokenizer: bool = False,
                 is_remove_header: bool = True):
        super(NLIDataset, self).__init__()

        # read data from data_path and ignore header row
        csv_file = open(data_path)
        reader = csv.reader(csv_file)
        texts = [item for item in reader]
        if is_remove_header:
            texts = texts[1:]

        if use_vi_tokenizer:
            logger.info("USE VI TOKENIZER: True")
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes - 2) as pool:
                texts = pool.map(process_multiple_line, texts)

        self.sentences = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        sentences = self.sentences[index]
        seqs = self.tokenizer(sentences,
                              padding='max_length',
                              max_length=self.max_seq_len,
                              truncation=True,
                              return_tensors="pt")
        return seqs

    def __len__(self):
        return len(self.sentences)


class NLI_CLS_Dataset(Dataset):
    """
    File: .json
    Data Structure: sentence1, sentence1, gold_label, annotator_label
    """

    def __init__(self,
                 data_path: str = None,
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_len: int = 256,
                 use_vi_tokenizer: bool = False):
        super(NLI_CLS_Dataset, self).__init__()

        # read data from data_path
        data = read_json(data_path)
        text1 = [d["sentence1"] for d in data]
        text2 = [d["sentence2"] for d in data]

        if use_vi_tokenizer:
            logger.info("USE VI TOKENIZER: True")
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes - 2) as pool:
                text1 = pool.map(process_single_line, text1)
                text2 = pool.map(process_single_line, text2)

        self.sentence1 = text1
        self.sentence2 = text2
        self.gold_label = [d["gold_label"] for d in data]
        self.annotator_label = [d["annotator_label"] for d in data]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        text1 = self.sentence1[index]
        text2 = self.sentence2[index]
        label = self.annotator_label[index]

        seqs = self.tokenizer(text1, text2,
                              padding='max_length',
                              max_length=self.max_seq_len,
                              truncation='only_second',
                              return_tensors="pt")
        for k in seqs:
            seqs[k] = seqs[k].squeeze(0)
        return seqs, torch.tensor(label)

    def __len__(self):
        return len(self.sentence1)

    def get_label_classes(self):
        return np.unique(self.gold_label, return_counts=True)

    def get_num_classes(self) -> int:
        return len(set(self.gold_label))

    def compute_class_weights(self) -> List:
        unique_data = np.unique(self.annotator_label)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_data, y=self.annotator_label)
        return list(class_weights)
