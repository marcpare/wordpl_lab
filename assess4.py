import random
import os
import json
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
from scipy.stats import beta
from lib import word_lists, clues

# epsilon = 12.35
# ws1 = "trace"
# strategy  = json.load(open("e4/e4.json", "r"))
# strategy = [s[0] for s in strategy]
# (0.491725768321513, (0.45808472610401146, 0.525403867261636))

# epsilon = 28.0
# ws1 = "salet"
# strategy  = json.load(open("d95/d95.json", "r"))
# (3679, 0.9589562381081815, (0.9523138123743408, 0.9651262243210579))

# epsilon = 27.0
# ws1 = "salet"
# strategy  = json.load(open("d95/d95.json", "r"))
# (1370, 0.9554744525547445, (0.9439425366988602, 0.965748040607082))

epsilon = 12.35
ws1 = "salet"
strategy  = json.load(open("d95/d95.json", "r"))



print("Assessing epsilon =", epsilon)
w1 = word_lists.valid_words.index(ws1)
pd = clues.pd(epsilon)
cwa = clues.cwa()

#############################################################################################################
#
# Search for best third guess in a four guess strategy
#
#############################################################################################################

memo_path = "d95/d95_third_guess_memo.json"
if not os.path.exists(memo_path):
    with open(memo_path, "w") as f:
        json.dump({}, f)
third_guess_memo = json.load(open(memo_path, "r"))

def best_third_guess(w1, c1, w2, c2):

    # global third_guess_memo

    # memo_key = f"{w1} {c1} {w2} {c2}"
    # if memo_key in third_guess_memo:
    #     return third_guess_memo[memo_key]

    ps1 = pd[c1][cwa[w1]]
    ps2 = pd[c2][cwa[w2]]
    ps = ps1 * ps2
    ps = ps[:, np.newaxis]

    best_w3 = None
    best_expected_wins = 0

    for w3 in range(word_lists.NVW):

        ps3 = pd[:][cwa[w3]]
        ps4 = ps * ps3
        expected_wins = np.sum(np.max(ps4, axis=0))

        if expected_wins >= best_expected_wins:
            best_expected_wins = expected_wins
            best_w3 = w3

    # with open(memo_path, "r") as f:
    #     third_guess_memo = json.load(f)
    #     third_guess_memo[memo_key] = best_w3
    # with open(memo_path, "w") as f:
    #     json.dump(third_guess_memo, f)

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


def worker_fn(results_queue):
    shuffled_answers = random.sample(word_lists.answers, len(word_lists.answers))

    for a in shuffled_answers:
        c1 = noisy_clue(ws1, a)
        w2 = strategy[c1]
        ws2 = word_lists.valid_words[w2]
        c2 = noisy_clue(ws2, a)

        # compute ws3
        w3 = best_third_guess(w1, c1, w2, c2)
        ws3 = word_lists.valid_words[w3]
        c3 = noisy_clue(ws3, a)

        w4 = best_final_guess(w1, c1, w2, c2, w3, c3)
        ws4 = word_lists.answers[w4]

        path = f"{ws1} -> {clues.clue_to_str(c1)} -> {ws2} -> {clues.clue_to_str(c2)} -> {ws3} -> {clues.clue_to_str(c3)} -> {ws4}"

        g = ws4

        if g == a:
            results_queue.put((path, 1))
        else:
            results_queue.put((path, 0))


class BayesianStrategyEstimator:
    def __init__(self, alpha=1, beta=1):
        # Initialize prior as Beta(1, 1) (uniform prior) if no prior knowledge
        self.alpha = alpha
        self.beta = beta

    def update(self, win):
        # Update the posterior with a win (1) or loss (0)
        if win:
            self.alpha += 1
        else:
            self.beta += 1

    def get_estimate(self):
        # Posterior mean and 95% credible interval
        mean = self.alpha / (self.alpha + self.beta)
        ci_low, ci_high = beta.ppf([0.025, 0.975], self.alpha, self.beta)
        return self.alpha + self.beta, mean, (float(ci_low), float(ci_high))

if __name__ == "__main__":

    estimator = BayesianStrategyEstimator()

    results_queue = Queue()

    workers = []

    for _ in range(mp.cpu_count() // 2):
        worker = Process(target=worker_fn, args=(results_queue,))
        worker.start()
        workers.append(worker)

    while True:
        path, result = results_queue.get()
        if result == 1:
            estimator.update(1)
            print(path, "WIN", estimator.get_estimate())
        else:
            estimator.update(0)
            print(path, "LOSE", estimator.get_estimate())
