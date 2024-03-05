from __future__ import annotations

import re

from pyvi import ViTokenizer

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


def normalize_encode(text: str) -> str:
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


def is_valid_vietnam_word(word: str) -> bool:
    """
    Check if a word is a valid word in the Vietnamese language.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is a valid Vietnamese word, False otherwise.

    """
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


def normalize_word_diacritic(word: str) -> str:
    """
    diacritic: á, à, ạ, ả, ã
    params:
        raw word
    return:
        word normalize
    """
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    diacritic = 0
    vowel_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = vowel_to_idx.get(char, (-1, -1))
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
            chars[index] = vowel[x][0]
        if not qu_or_gi or index != 1:
            vowel_index.append(index)
    if len(vowel_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_to_idx.get(chars[1])
                chars[1] = vowel[x][diacritic]
            else:
                x, y = vowel_to_idx.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel[x][diacritic]
                else:
                    chars[1] = vowel[5][diacritic] if chars[1] == 'i' else vowel[9][diacritic]
            return ''.join(chars)
        return word

    for index in vowel_index:
        x, y = vowel_to_idx[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel[x][diacritic]
            # for index2 in vowel_index:
            #     if index2 != index:
            #         x, y = vowel_to_idx[chars[index]]
            #         chars[index2] = vowel[x][0]
            return ''.join(chars)

    if len(vowel_index) == 2:
        if vowel_index[-1] == len(chars) - 1:
            x, y = vowel_to_idx[chars[vowel_index[0]]]
            chars[vowel_index[0]] = vowel[x][diacritic]
            # x, y = vowel_to_idx[chars[vowel_index[1]]]
            # chars[vowel_index[1]] = vowel[x][0]
        else:
            # x, y = vowel_to_idx[chars[vowel_index[0]]]
            # chars[vowel_index[0]] = vowel[x][0]
            x, y = vowel_to_idx[chars[vowel_index[1]]]
            chars[vowel_index[1]] = vowel[x][diacritic]
    else:
        # x, y = vowel_to_idx[chars[vowel_index[0]]]
        # chars[vowel_index[0]] = vowel[x][0]
        x, y = vowel_to_idx[chars[vowel_index[1]]]
        chars[vowel_index[1]] = vowel[x][diacritic]
        # x, y = vowel_to_idx[chars[vowel_index[2]]]
        # chars[vowel_index[2]] = vowel[x][0]
    return ''.join(chars)


def remove_punctuation(text: str) -> str:
    """
    Removes punctuation characters from the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with all punctuation characters removed.

    """
    text = re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~]", '', text)
    return text


def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by applying several normalization and tokenization steps.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.

    """
    text = normalize_encode(normalize_word_diacritic(text))
    text = text.lower()
    text = remove_punctuation(text)
    text = ViTokenizer.tokenize(text)
    return text
