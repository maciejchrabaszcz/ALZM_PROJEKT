from tqdm import tqdm
import numpy as np
import re
import random
from nltk import bigrams
from collections import Counter


def remove_unknown_symbols(known_symbols, text):
    pattern = "[^" + known_symbols + "]"
    return re.sub(pattern, "", text)


def create_empty_dict(known_chars):
    empty_dict = {}
    for i in known_chars:
        for j in known_chars:
            empty_dict.setdefault(i, {})[j] = 0
    return empty_dict


def create_count_dict(text, known_chars):
    count_dict = create_empty_dict(known_chars)
    counted = dict(Counter(bigrams(text)))
    for chars, count in tqdm(counted.items()):
        char_a = chars[0]
        char_b = chars[1]
        if char_a in known_chars and char_b in known_chars:
            count_dict[char_a][char_b] = count
    return count_dict


def create_perc_dict(count_dict, known_chars):
    perc_dict = create_empty_dict(known_chars)
    for i in known_chars:
        total_count = sum(count_dict[i].values())
        for j in known_chars:
            letter_perc = (count_dict[i][j] + 1) / total_count
            perc_dict[i][j] = np.log(letter_perc)
    return perc_dict


def create_rand_crypt(known_chars):
    list_chars = list(known_chars)
    random.shuffle(list_chars)
    rand_chars = "".join(list_chars)
    return rand_chars
