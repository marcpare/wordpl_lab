import os
import json
import random
import math
from math import exp
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from numba import njit, prange

from lib import clues, word_lists


ws1 = "trace"
w1 = word_lists.valid_words.index(ws1)

epsilon = 12.35

pd = clues.pd(epsilon)
cwa = clues.cwa()


@njit(parallel=True)
def highest_entropy_guesses(w1, c1, n):
    word_entropies = np.zeros(word_lists.NVW)
    for w2 in prange(word_lists.NVW):
        ps = np.zeros(3**5)
        for c2 in range(3**5):
            for a in range(word_lists.NAW):
                ps[c2] += pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]
        ps = ps / np.sum(ps)
        word_entropies[w2] += -np.sum(ps * np.log2(ps))

    word_entropies = word_entropies.argsort()
    return np.flip(word_entropies[-n:])


def highest_entropy_guess_with_lookahead(w1, c1):
    best_w2_wins = 0
    best_w2 = None
    for w2 in highest_entropy_guesses(w1, c1, 1):
        total_wins = 0
        for c2 in tqdm(range(3**5)):

            # likelihood of answer a given w1, c1, w2, c2
            lia = np.zeros(word_lists.NAW)
            for a in range(word_lists.NAW):
                lia[a] = pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]

            slia = np.argsort(lia)
            possible_answers = slia[-int(word_lists.NAW / 50):]

            w3, ew = best_third_guess(w1, c1, w2, c2, possible_answers)
            # w32, ew2 = best_third_guess(w1, c1, w2, c2, np.array(range(word_lists.NAW)))

            # gather distinct characters in possible answers
            distinct_chars = set()
            for a in slia[-int(word_lists.NAW / 50):]:
                for c in word_lists.answers[a]:
                    distinct_chars.add(c)

            # remove characters already guess in w1 and w2
            distinct_chars = distinct_chars - set(word_lists.valid_words[w1]) - set(word_lists.valid_words[w2])

            # how many of these characters does w3 contain?
            w3_chars = set(word_lists.valid_words[w3])
            print("char overlap", len(distinct_chars & w3_chars))
            print("")

            # count char overlap of every possible w3
            overlap_counts = np.zeros(6)
            for w3 in range(word_lists.NVW):
                w3_chars = set(word_lists.valid_words[w3])
                overlap = len(distinct_chars & w3_chars)
                overlap_counts[overlap] += 1

            print(overlap_counts)

            # print(w3, w32)
            break

            total_wins += ew

        print(word_lists.valid_words[w2], total_wins)

        if total_wins > best_w2_wins:
            best_w2_wins = total_wins
            best_w2 = w2

@njit(parallel=True)
def best_third_guess(w1, c1, w2, c2, possible_answers):

    expected_wins = np.zeros(word_lists.NVW)

    for w3 in prange(word_lists.NVW):
        total_expected_wins = 0
        for c3 in range(3**5):
            bew = 0
            for a in possible_answers:
                bew = max(bew, pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]] * pd[c3][cwa[w3][a]])
            total_expected_wins += bew
        expected_wins[w3] = total_expected_wins

    best_w3 = np.argmax(expected_wins)
    most_expected_wins = expected_wins[best_w3]

    return (best_w3, most_expected_wins)


if __name__ == "__main__":

    for c1 in tqdm(range(3**5)):
        w2 = highest_entropy_guess_with_lookahead(w1, c1)
        print(c1, w2)
        
        # trice 0.026712219801804816
        # trice 1.2782270858146572