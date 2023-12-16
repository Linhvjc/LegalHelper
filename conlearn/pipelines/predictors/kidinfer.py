from __future__ import annotations

import json
import re

import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

from .base import InferenceModel


class Inference(InferenceModel):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    def __init__(
        self,
        name_of_model_path: str,
        corpus_path: str,
        max_seq_len_query: int = 64,
        max_seq_len_doc: int = 256,
        batch_size: int = 32,
        use_lowercase: bool = True,
        use_remove_punc: bool = True,
        pooler_type: str = 'avg',
    ) -> None:
        self.load_model_and_tokenizer(name_of_model_path)

        self.batch_size = batch_size
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_doc = max_seq_len_doc
        self.use_lowercase = use_lowercase
        self.use_remove_punc = use_remove_punc
        self.pooler_type = pooler_type

        self.normalize_text = NormalizeText()

        # indexing
        corpus = self.load_json(corpus_path)
        self.corpus_texts, self.corpus_ids = [], []
        for doc in corpus:
            self.corpus_texts.append(doc['text'])
            self.corpus_ids.append(doc['meta']['id'])
        self.corpus_embedding = self.generate_embedding(
            self.corpus_texts, self.max_seq_len_doc,
        )

    @staticmethod
    def load_json(filename: str) -> dict:
        with open(filename, encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_model_and_tokenizer(self, name_of_model_path):
        self.model = AutoModel.from_pretrained(
            name_of_model_path,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name_of_model_path)

    def preprocess_input(
        self,
        text: str,
        max_seq_len: int = 256,
        special_tokens_count: int = 2,
    ) -> tuple[list[int], list[int]]:
        """
        Tokenizes and converts the input text into feature vectors suitable for model input.

        Args:
            text (str): The input text to be converted into features.
            tokenizer (PreTrainedTokenizer): The pre-trained tokenizer object to tokenize the input text.
            max_seq_len (int, optional): Maximum sequence length for the output feature vectors.
                Defaults to 128.
            special_tokens_count (int, optional): Number of special tokens like [CLS] and [SEP] in the tokenizer vocabulary.
                Defaults to 2.
            lower_case (bool, optional): Whether to convert the text to lowercase. Defaults to False.
            remove_punc (bool, optional): Whether to remove punctuation from the text. Defaults to False.
            **kwargs: Additional keyword arguments for future expansion.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing two lists - input_ids and attention_mask.
            - input_ids (List[int]): List of token IDs representing the input text with padding if necessary.
            - attention_mask (List[int]): List of attention mask values (1 for real tokens, 0 for padding tokens).

        Raises:
            AssertionError: If the length of input_ids does not match the specified max_seq_len.

        Example:
            text = "Example input text for tokenization."
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            input_ids, attention_mask = convert_text_to_features(text, tokenizer)
        """

        unk_token = self.tokenizer.unk_token

        cls_token = self.tokenizer.cls_token

        sep_token = self.tokenizer.sep_token

        pad_token_id = self.tokenizer.pad_token_id

        # Normalize text
        text = self.normalize_text.process(
            text=text,
            use_lowercase=self.use_lowercase,
            use_remove_punc=self.use_remove_punc,
        )

        texts = text.split()  # Some are spaced twice

        tokens = []
        # Tokenizer
        for word in texts:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Truncate data
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]

        # Add [CLS] token
        tokens = [cls_token] + tokens

        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # TODO use smart padding in here
        # Zero-pad up to the sequence length. This is static method padding
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)

        assert len(input_ids) == max_seq_len, 'Error with input length {} vs {}'.format(
            len(input_ids),
            max_seq_len,
        )

        return (input_ids, attention_mask)

    def generate_embedding(self, texts: list[str], max_seq_len: int = 256) -> torch.Tensor:
        texts_embedding = None

        for i in range(0, len(texts), self.batch_size):
            batch_input_ids, batch_attention_mask = [], []
            for text in texts[i: i + self.batch_size]:
                (input_ids, attention_mask) = self.preprocess_input(
                    text=text,
                    max_seq_len=max_seq_len,
                )
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)

            batch_input_ids = torch.tensor(
                batch_input_ids, dtype=torch.long,
            ).to(self.model.device)
            batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(
                self.model.device,
            )
            self.model.eval()
            with torch.no_grad():
                inputs = {
                    'input_ids': batch_input_ids,
                    'attention_mask': batch_attention_mask,
                }

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                if self.pooler_type in ['cls_before_pooler', 'cls']:
                    embedding = last_hidden[:, 0]
                elif self.pooler_type == 'avg':
                    embedding = (last_hidden * batch_attention_mask.unsqueeze(-1)).sum(
                        1,
                    ) / batch_attention_mask.sum(-1).unsqueeze(-1)
            if texts_embedding is None:
                texts_embedding = embedding.detach().cpu().numpy()
            else:
                texts_embedding = np.append(
                    texts_embedding,
                    embedding.detach().cpu().numpy(),
                    axis=0,
                )

        return texts_embedding

    def perform_inference(self, queries: str | list[str], top_k: int = 5) -> list[dict]:
        if isinstance(queries, str):
            queries = [queries]

        queries_embedding = self.generate_embedding(
            queries, self.max_seq_len_query,
        )

        scores = queries_embedding @ self.corpus_embedding.T

        preds = []
        for query, score in zip(queries, scores):
            ind = np.argpartition(score, -top_k)[-top_k:]
            pred = list(map(self.corpus_ids.__getitem__, ind))
            preds.append({'query': query, 'response': pred})

        return preds

    def postprocess_output(self, output):
        pass


class NormalizeText:
    def __init__(self):
        self.vowel = [
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
        self.vowel_to_idx = {}
        for i in range(len(self.vowel)):
            for j in range(len(self.vowel[i]) - 1):
                self.vowel_to_idx[self.vowel[i][j]] = (i, j)

    def process(
        self,
        text: str,
        use_lowercase: bool = False,
        use_remove_punc: bool = False,
    ) -> str:
        text = self.normalize_encode(self.normalize_word_diacritic(text))

        if use_lowercase:
            text = text.lower()
        if use_remove_punc:
            text = self.remove_punctuation(text)
        return text

    def remove_punctuation(self, text: str) -> str:
        text = re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~]", '', text)
        return text

    def is_valid_vietnam_word(self, word: str) -> bool:
        chars = list(word)
        vowel_index = -1
        for index, char in enumerate(chars):
            x, _ = self.vowel_to_idx.get(char, (-1, -1))
            if x != -1:
                if vowel_index == -1:
                    vowel_index = index
                else:
                    if index - vowel_index != 1:
                        return False
                    vowel_index = index
        return True

    def normalize_word_diacritic(self, word: str) -> str:
        """
        diacritic: á, à, ạ, ả, ã
        params:
            raw word
        return:
            word normalize
        """
        if not self.is_valid_vietnam_word(word):
            return word

        chars = list(word)
        diacritic = 0
        vowel_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.vowel_to_idx.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                diacritic = y
                chars[index] = self.vowel[x][0]
            if not qu_or_gi or index != 1:
                vowel_index.append(index)
        if len(vowel_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.vowel_to_idx.get(chars[1])
                    chars[1] = self.vowel[x][diacritic]
                else:
                    x, y = self.vowel_to_idx.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.vowel[x][diacritic]
                    else:
                        chars[1] = (
                            self.vowel[5][diacritic]
                            if chars[1] == 'i'
                            else self.vowel[9][diacritic]
                        )
                return ''.join(chars)
            return word

        for index in vowel_index:
            x, y = self.vowel_to_idx[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.vowel[x][diacritic]
                # for index2 in vowel_index:
                #     if index2 != index:
                #         x, y = vowel_to_idx[chars[index]]
                #         chars[index2] = vowel[x][0]
                return ''.join(chars)

        if len(vowel_index) == 2:
            if vowel_index[-1] == len(chars) - 1:
                x, y = self.vowel_to_idx[chars[vowel_index[0]]]
                chars[vowel_index[0]] = self.vowel[x][diacritic]
                # x, y = vowel_to_idx[chars[vowel_index[1]]]
                # chars[vowel_index[1]] = vowel[x][0]
            else:
                # x, y = vowel_to_idx[chars[vowel_index[0]]]
                # chars[vowel_index[0]] = vowel[x][0]
                x, y = self.vowel_to_idx[chars[vowel_index[1]]]
                chars[vowel_index[1]] = self.vowel[x][diacritic]
        else:
            # x, y = vowel_to_idx[chars[vowel_index[0]]]
            # chars[vowel_index[0]] = vowel[x][0]
            x, y = self.vowel_to_idx[chars[vowel_index[1]]]
            chars[vowel_index[1]] = self.vowel[x][diacritic]
            # x, y = vowel_to_idx[chars[vowel_index[2]]]
            # chars[vowel_index[2]] = vowel[x][0]
        return ''.join(chars)

    def normalize_encode(self, text: str) -> str:
        """
        normalize unicode encoding
        params:
            raw text
        return:
            normalization text
        """
        dicchar = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|',
        )
        charutf8 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|',
        )
        for i in range(len(char1252)):
            dicchar[char1252[i]] = charutf8[i]

        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: dicchar[x.group()],
            text,
        )
