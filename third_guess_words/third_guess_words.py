import random
import json
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
from scipy.stats import beta
from lib import word_lists, clues

epsilon = 12.35

ws1 = "trace"
w1 = word_lists.valid_words.index(ws1)

strategy  = json.load(open("e4/e4.json", "r"))
strategy = [s[0] for s in strategy]

pd = clues.pd(epsilon)
cwa = clues.cwa()

#############################################################################################################
#
# Search for best third guess in a four guess strategy
#
#############################################################################################################

third_guess_memo = {}

def best_third_guess(w1, c1, w2, c2):

    memo_key = (w1, c1, w2, c2)
    if memo_key in third_guess_memo:
        return third_guess_memo[memo_key]

    ps1 = pd[c1][cwa[w1]]
    ps2 = pd[c2][cwa[w2]]
    ps = ps1 * ps2
    ps = ps[:, np.newaxis]

    best_w3 = None
    best_expected_wins = 0

    # What are the most possible expected wins?
    print(ps.shape)

    print(np.sum(ps))

    for w3 in range(word_lists.NVW):

        ps3 = pd[:][cwa[w3]]
        ps4 = ps * ps3
        expected_wins = np.sum(np.max(ps4, axis=0))

        if expected_wins >= best_expected_wins:
            print(best_expected_wins)
            best_expected_wins = expected_wins
            best_w3 = w3

    print("fraction", best_expected_wins / np.sum(ps))

    third_guess_memo[memo_key] = best_w3
    return best_w3

#############################################################################################################
#
# Make an optimal final guess
#
#############################################################################################################

def best_final_guess(w1, c1, w2, c2, w3, c3):

    ps1 = pd[c1][cwa[w1]]
    ps2 = pd[c2][cwa[w2]]
    ps3 = pd[c3][cwa[w3]]
    ps4 = ps1 * ps2 * ps3
    ps4 = ps4[:, np.newaxis]

    return np.argmax(ps4)


def noisy_clue(guess, answer):
    real_clues = ''
    for (i, letter) in enumerate(guess):
        if letter == answer[i]:
            real_clues += '0'
        elif letter in answer:
            real_clues += '1'
        else:
            real_clues += '2'
    noisy_clues = ''
    for a in real_clues:
        # Choose randomly with probability 3/(2+e^(ε/5)); equivalent to choosing
        # randomly *among the incorrect options* with probability 2/(2+e^(ε/5)).
        if random.random() < 3./(2.+math.exp(epsilon/5)):
            noisy_clues += random.choice(['0', '1', '2'])
        else:
            noisy_clues += a

    return int(noisy_clues, 3)


def run_trial():
    shuffled_answers = random.sample(word_lists.answers, len(word_lists.answers))

    for a in shuffled_answers:
        c1 = noisy_clue(ws1, a)
        w2 = strategy[c1]
        ws2 = word_lists.valid_words[w2]
        c2 = noisy_clue(ws2, a)

        # compute ws3
        w3 = best_third_guess(w1, c1, w2, c2)

        break



if __name__ == "__main__":

    run_trial()