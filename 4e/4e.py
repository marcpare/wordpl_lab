import os
import json
import random
import math
from math import exp
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from lib import clues, word_lists

ws1 = "trace"
w1 = word_lists.valid_words.index(ws1)

epsilon = 12.35

pd = clues.pd(epsilon)
cwa = clues.cwa()


def highest_entropy_guess_with_lookahead(w1, c1):
    word_entropies = []

    good_words_indices = word_lists.good_word_indices
    psp = pd[c1][cwa[w1]]
    psp = psp[:, np.newaxis]

    for w in range(word_lists.NVW):
        ps = np.sum(psp * pd[:][cwa[w]], axis=0)
        ps = ps / np.sum(ps) # normalize
        entropy = -np.sum(ps * np.log2(ps))
        word_entropies.append((w, entropy))

    word_entropies.sort(key=lambda x: x[1], reverse=True)

    print(word_entropies[:10])
    return word_entropies[0][0]

    best_w2 = None
    best_w2_wins = 0
    for w2, _ in word_entropies[:10]:

        # compute expected wins for each with perfect third and fourth guesses
        total_w2_wins = 0

        ps1 = pd[c1][cwa[w1]]

        for c2 in tqdm(range(3**5)):

            ps2 = pd[c2][cwa[w2]]
            ps = ps1 * ps2
            ps = ps[:, np.newaxis]

            # find the best words for a third guess
            best_expected_wins = 0

            for w3 in good_words_indices:

                ps3 = pd[:][cwa[w3]]
                ps4 = ps * ps3
                expected_wins = np.sum(np.max(ps4, axis=0))
                if expected_wins >= best_expected_wins:
                    best_expected_wins = expected_wins

            total_w2_wins += best_expected_wins

        if total_w2_wins > best_w2_wins:
            best_w2_wins = total_w2_wins
            best_w2 = w2

        print(PCS.valid_words[w2], total_w2_wins)

    return (w1, c1, best_w2)



def best_third_guess(args):

    w1, c1, w2, c2 = args

    PCS = Precomputations()
    pd = PCS.pd
    cwa = PCS.cwa
    good_words_indices = PCS.good_words_indices

    ps1 = pd[c1][cwa[w1]]
    ps2 = pd[c2][cwa[w2]]
    ps = ps1 * ps2
    ps = ps[:, np.newaxis]

    best_w3 = None
    best_expected_wins = 0

    for w3 in good_words_indices:

        ps3 = pd[:][cwa[w3]]
        ps4 = ps * ps3
        expected_wins = np.sum(np.max(ps4, axis=0))

        if expected_wins >= best_expected_wins:
            best_expected_wins = expected_wins
            best_w3 = w3

    return (best_w3, best_expected_wins)


if __name__ == "__main__":

    for c1 in tqdm(range(3**5)):
        w2 = highest_entropy_guess_with_lookahead(w1, c1)
        

    

    # maximize expected wins for turn 3 strategy
    # for c1 in range(3**5):
    #     if len(strategy[c1]) > 1:
    #         continue

    #     w2 = strategy[c1][0]

    #     tasks = [(w1, c1, w2, c2) for c2 in range(3**5)]

    #     results = {}

    #     with mp.Pool(mp.cpu_count() // 2) as pool:
    #         for result in tqdm(pool.imap_unordered(best_third_guess, tasks), total=len(tasks)):
    #             w1, c1, w2, c2, w3 = result
    #             results[c2] = w3

    #     for c2 in range(3**5):
    #         strategy[c1].append(results[c2])

    #     with open("4e_strategy.json", "w") as f:
    #         json.dump(strategy, f)


