"""

Compute optimal second guess strategy for WORDPL by exhaustive search.

Exhaustive search is trickier in WORDPL than traditional wordle because it is possible
for any clue to be given at any turn. Therefore every possible clue must be considered
at each level of the search:

```
    for every possible first clue:
        for every possible second guess:
            for every possible second clue:
                for every possible third guess:
                    for every possible third clue:
                        compute the expected number of wins
```

Two pruning steps make this feasible:

1. Only consider the top 10 second guesses with the highest clue entropy.
2. Prune possible answers to the top 2% most likely answers after the second guess.

"""

import numpy as np
from numba import njit, prange

from lib import clues, word_lists

ws1 = "salet"
w1 = word_lists.valid_words.index(ws1)

epsilon = 27.0

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
    for w2 in highest_entropy_guesses(w1, c1, 10):
        total_wins = 0
        for c2 in range(3**5):

            # likelihood of answer a given w1, c1, w2, c2
            lia = np.zeros(word_lists.NAW)
            for a in range(word_lists.NAW):
                lia[a] = pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]

            slia = np.argsort(lia)
            possible_answers = slia[-int(word_lists.NAW / 50):]

            w3, ew = best_third_guess(w1, c1, w2, c2, possible_answers)

            total_wins += ew

        # print(word_lists.valid_words[w2], total_wins)

        if total_wins > best_w2_wins:
            best_w2_wins = total_wins
            best_w2 = w2

    return best_w2

@njit(parallel=True)
def best_third_guess(w1, c1, w2, c2):

    # likelihood of answer a given w1, c1, w2, c2
    lia = np.zeros(word_lists.NAW)
    for a in range(word_lists.NAW):
        lia[a] = pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]]
    slia = np.argsort(lia)

    # filter the possible answers to the top 2%
    possible_answers = slia[-int(word_lists.NAW / 50):]

    expected_wins = np.zeros(word_lists.NVW)

    for w3 in prange(word_lists.NVW):
        total_expected_wins = 0
        for c3 in range(3**5):
            bew = 0
            for a in possible_answers:
                bew = max(bew, pd[c1][cwa[w1][a]] * pd[c2][cwa[w2][a]] * pd[c3][cwa[w3][a]])
            total_expected_wins += bew
        expected_wins[w3] = total_expected_wins

    return np.argmax(expected_wins)


if __name__ == "__main__":

    for c1 in range(3**5):
        w2 = highest_entropy_guess_with_lookahead(w1, c1)
        print(c1, w2)