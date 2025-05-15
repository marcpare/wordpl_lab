import numpy as np
from tqdm import tqdm
from multiprocessing import Queue, Process
import multiprocessing as mp

from lib import clues, word_lists

from numba import njit, prange

ws1 = "trace"
w1 = word_lists.valid_words.index(ws1)
epsilon1 = 9.3
epsilon2 = 5.3

pd1 = clues.pd(epsilon1)
pd2 = clues.pd(epsilon2)

cwa = clues.cwa()

@njit(parallel=True)
def best_second_guess(w1, c1):

    expected_wins = np.zeros(word_lists.NVW)

    for w2 in prange(word_lists.NVW):
        total_expected_wins = 0
        for c2 in range(3**5):
            bew = 0
            for a in range(word_lists.NAW):
                bew = max(bew, pd1[c1][cwa[w1][a]] * pd2[c2][cwa[w2][a]])
            total_expected_wins += bew
        expected_wins[w2] = total_expected_wins

    best_w2 = np.argmax(expected_wins)
    most_expected_wins = expected_wins[best_w2]

    return best_w2, most_expected_wins



if __name__ == "__main__":

    total_expected_wins = 0
    for c1 in tqdm(range(3**5)):
        ws2, expected_wins = best_second_guess(w1, c1)
        total_expected_wins += expected_wins
        print(f"{clues.clue_to_str(c1)} -> {ws2} (expected wins: {expected_wins})", total_expected_wins)

    print("done")

