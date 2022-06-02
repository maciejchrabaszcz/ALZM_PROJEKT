import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming
from nltk import bigrams
from collections import Counter


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


def score_likelihood(decrypted_text, perc_dict):
    total_likelihood = 0
    counted = dict(Counter(bigrams(decrypted_text)))
    for chars, count in counted.items():
        pair_likelihood = count*perc_dict[chars[0]][chars[1]]
        total_likelihood += pair_likelihood
    return total_likelihood


def shuffle_pair(current_dict):
    a, b = random.sample(current_dict.keys(), 2)
    proposed_dict = current_dict.copy()
    proposed_dict[a], proposed_dict[b] = proposed_dict[b], proposed_dict[a]
    return proposed_dict


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
    iters=1e+5,
    verbose=False,
    num_steps_to_eval: int = 1000,
    real_text=None,
):
    best_score = []
    best_text = []
    hamming_losses = []
    scores = []

    # Step 1 - Create a random decryption key
    current_dict = str_to_key(known_chars, crypt_keys)
    # Step 2 - Decrypt the text
    current_decrypted = apply_dict(cyphered_text, current_dict)
    # Step 3 - Score the (log) likelihood of the decrypted text
    current_score = score_likelihood(current_decrypted, perc_dict)
    scores.append(current_score)

    for i in tqdm(range(iters), leave=False):
        # Step 4 - Randomly shuffle two letter pairings
        proposed_dict = shuffle_pair(current_dict)
        # Step 5 - Decrypt the text again with the new key
        proposed_decrypted = apply_dict(cyphered_text, proposed_dict)
        # Step 6 - Recompute the log-likelihood score
        proposed_score = score_likelihood(proposed_decrypted, perc_dict)
        # Step 7 - Evaluate the difference
        # If the likelihood has improved, accept the new key
        # If the likelihood hasn't improved but the value exceeds a randomly drawn value between 0-1, also accept the new key
        # Otherwise, reject the new key
        if eval_proposal(proposed_score, current_score):
            current_dict = proposed_dict
            current_score = proposed_score
            current_decrypted = proposed_decrypted
            scores.append(current_score)
        # Repeat the above for the given amount of iterations

        if i % num_steps_to_eval == 0:
            best_score.append(current_score)
            best_text.append(current_decrypted)
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
