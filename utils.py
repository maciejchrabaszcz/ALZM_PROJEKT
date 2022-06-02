from tqdm import tqdm
import numpy as np
import re
import random
from nltk import ngrams
from collections import Counter
from typing import Dict
from itertools import product


def remove_unknown_symbols(known_symbols, text):
    pattern = "[^" + known_symbols + "]"
    return re.sub(pattern, "", text)


def create_empty_dict(known_chars, num_previous_chars: int = 1):
    empty_dict: Dict[str, Dict[str, int]] = {}
    for prev_chars in product(
        known_chars, repeat=num_previous_chars
    ):
        for j in known_chars:
            empty_dict.setdefault("".join(prev_chars), {})[j] = 0
    return empty_dict


def create_count_dict(text, known_chars, num_previous_chars: int = 1):
    count_dict = create_empty_dict(known_chars, num_previous_chars)
    counted = dict(Counter(ngrams(text, num_previous_chars + 1)))
    for chars, count in tqdm(counted.items()):
        prev_chars = "".join(chars[:-1])
        next_char = chars[-1]
        # if char_a in known_chars and char_b in known_chars:
        count_dict.setdefault(prev_chars, {})[next_char] = count
    return count_dict


def create_perc_dict(count_dict, known_chars, num_previous_chars: int = 1):
    perc_dict = create_empty_dict(known_chars, num_previous_chars)
    for prev_chars, counts_dict in count_dict.items():
        total_count = sum(counts_dict.values())
        for char in known_chars:
            if total_count > 0:
                letter_perc = (counts_dict.get(char, 0) + 1) / total_count
            else:
                letter_perc = 1e-24
            perc_dict[prev_chars][char] = np.log(letter_perc)
    return perc_dict


def create_rand_crypt(known_chars):
    list_chars = list(known_chars)
    random.shuffle(list_chars)
    rand_chars = "".join(list_chars)
    return rand_chars
