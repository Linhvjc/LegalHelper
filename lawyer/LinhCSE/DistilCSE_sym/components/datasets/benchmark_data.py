from typing import Text, List, Dict

from pyvi import ViTokenizer

from utils.io import read_json


class BenchmarkData:
    def __init__(self, query_path: str, corpus_path: str, use_vi_tokenizer: bool = False):
        self.query_data = read_json(query_path)
        self.corpus_data = read_json(corpus_path)
        self.use_vi_tokenizer = use_vi_tokenizer

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: get_query(), get_corpus(), get_relevant_docs()
        :rtype:
        """
        return self.get_query(), self.get_corpus(), self.get_relevant_docs()

    def get_number_queries(self):
        return len(self.query_data)

    def get_number_corpus(self):
        return len(self.corpus_data)

    def get_query(self) -> Dict[Text, Text]:
        """
        :return: [{id1: text1}, {id2: text2}, ...]
        :rtype:
        """
        if self.use_vi_tokenizer:
            result = {q["id"]: ViTokenizer.tokenize(q["query"]) for q in self.query_data}
            return result
        result = {q["id"]: q["query"] for q in self.query_data}
        return result

    def get_relevant_docs(self) -> Dict[Text, List[Text]]:
        result = {q["id"]: set(q["gt"]) for q in self.query_data}
        return result

    def get_corpus(self) -> Dict[Text, Text]:
        """
        :return: [{id1: text1}, {id2: text2}, ...]
        :rtype:
        """
        if self.use_vi_tokenizer:
            result = {c["meta"]["id"]: ViTokenizer.tokenize(c["text"]) for c in self.corpus_data}
            return result
        result = {c["meta"]["id"]: c["text"] for c in self.corpus_data}
        return result
