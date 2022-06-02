import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming
from nltk import ngrams
from collections import Counter
from utils import create_rand_crypt


def str_to_key(known_chars, chars):
    crypt_dict = {}
    for i in range(len(known_chars)):
        crypt_dict[known_chars[i]] = chars[i]
    return crypt_dict


def apply_dict(text, key_dict):
    text_list = list(text)
    for i in range(len(text_list)):
        new_char = key_dict[text_list[i]]
        text_list[i] = new_char
    new_text = "".join(text_list)
    return new_text


def invert_dict(key_dict):
    inverted_dict = {v: k for k, v in key_dict.items()}
    return inverted_dict


def score_likelihood(decrypted_text, perc_dict, num_previous_chars: int = 1):
    total_likelihood = 0
    counted = dict(Counter(ngrams(decrypted_text, num_previous_chars + 1)))
    for chars, count in counted.items():
        pair_likelihood = count * perc_dict[chars[0]][chars[1]]
        total_likelihood += pair_likelihood
    return total_likelihood


def shuffle_pair(current_dict):
    a, b = random.sample(current_dict.keys(), 2)
    proposed_dict = current_dict.copy()
    proposed_dict[a], proposed_dict[b] = proposed_dict[b], proposed_dict[a]
    return proposed_dict


def shuffle_keys(current_dict, num_keys_to_shuffle: int = 2):
    if num_keys_to_shuffle == 2:
        return shuffle_pair(current_dict)

    keys_to_shuffle = random.sample(current_dict.keys(), num_keys_to_shuffle)
    values = [current_dict[key] for key in keys_to_shuffle]
    random.shuffle(values)
    proposed_dict = current_dict.copy()
    for key, value in zip(keys_to_shuffle, values):
        proposed_dict[key] = value
    return proposed_dict


def initiate_with_good_dict(
    cyphered_text,
    known_chars,
    perc_dict,
    num_tries=1000,
    num_previous_chars: int = 1,
):
    best_score = -np.Inf
    for _ in range(num_tries):
        current_crypt_keys = create_rand_crypt(known_chars=known_chars)
        current_dict = str_to_key(known_chars, current_crypt_keys)
        # Step 2 - Decrypt the text
        current_decrypted = apply_dict(cyphered_text, current_dict)
        # Step 3 - Score the (log) likelihood of the decrypted text
        current_score = score_likelihood(
            current_decrypted, perc_dict, num_previous_chars
        )
        if best_score > current_score:
            best_dict = current_dict
    return best_dict


def eval_proposal(proposed_score, current_score):
    diff = proposed_score - current_score
    diff = min(1, diff)
    diff = max(-1000, diff)
    ratio = np.exp(diff)
    if ratio >= 1 or ratio > np.random.uniform(0, 1):
        return True
    else:
        return False


def decrypt_MCMC(
    cyphered_text,
    perc_dict,
    crypt_keys,
    known_chars,
    iters=1e5,
    verbose=False,
    eval_every: int = 1000,
    real_text=None,
    num_previous_chars: int = 1,
    num_tries_to_initiate_dict=1,
):
    best_score = []
    best_text = []
    hamming_losses = []
    scores = []
    # Krok 1 - Utworzenie początkowego klucza deszyfrującego
    if num_tries_to_initiate_dict > 1:
        current_dict = initiate_with_good_dict(
            cyphered_text,
            known_chars,
            perc_dict,
            num_tries_to_initiate_dict,
            num_previous_chars,
        )
    current_dict = str_to_key(known_chars, crypt_keys)
    # Krok 2 - Odszyfrowanie tekstu
    current_decrypted = apply_dict(cyphered_text, current_dict)
    # Krok 3 - Ocena przy użyciu log wiarogodności na podstawie częstości
    current_score = score_likelihood(
        current_decrypted, perc_dict, num_previous_chars
    )
    scores.append(current_score)

    for i in tqdm(range(iters), leave=False):
        # Krok 4 - Losowa zmiana szyfru w dwóch znakach
        proposed_dict = shuffle_pair(current_dict)
        # Krok 5 - Odszyfrowanie z nowym kluczem
        proposed_decrypted = apply_dict(cyphered_text, proposed_dict)
        # Krok 6 - Obliczenie log wiarogodności
        proposed_score = score_likelihood(
            proposed_decrypted, perc_dict, num_previous_chars
        )
        # Krok 7 - Sprawdź nowy klucz licząc różnice w log wiarogodnościach
        # Jeżeli klucz lepszy to zaakcpetuj go
        # Jeżeli nie to wylosuj
        # Otherwise, reject the new key
        if eval_proposal(proposed_score, current_score):
            current_dict = proposed_dict
            current_score = proposed_score
            current_decrypted = proposed_decrypted

        if i % eval_every == 0:
            best_score.append(current_score)
            best_text.append(current_decrypted)
            scores.append(current_score)
            if real_text:
                hamming_losses.append(
                    hamming(list(real_text), list(current_decrypted))
                )

        if verbose == True and i % 1000 == 0:
            print(
                "Iteration: "
                + str(i)
                + ". Score: "
                + str(current_score)
                + ". Message: "
                + current_decrypted[0:70]
            )

    return current_dict, best_score, best_text, hamming_losses, scores
