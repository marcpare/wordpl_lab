import random
import json
import math
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
from scipy.stats import beta
from lib import word_lists, clues

ws1 = "trace"
w1 = word_lists.valid_words.index(ws1)

epsilon1 = 9.3
epsilon2 = 5.3

pd1 = clues.pd(epsilon1)
pd2 = clues.pd(epsilon2)

cwa = clues.cwa()

strategy  = json.load(open("g3/g3.json", "r"))

# (11577, 0.05493651204975382, (0.05085946210040406, 0.05915919742890638))


#############################################################################################################
#
# Make an optimal final guess
#
#############################################################################################################

def best_final_guess(w1, c1, w2, c2):

    ps1 = pd1[c1][cwa[w1]]
    ps2 = pd2[c2][cwa[w2]]
    ps3 = ps1 * ps2
    ps3 = ps3[:, np.newaxis]

    return np.argmax(ps3)


def noisy_clue(guess, answer, epsilon):
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
        c1 = noisy_clue(ws1, a, epsilon1)
        w2 = strategy[c1]
        ws2 = word_lists.valid_words[w2]
        c2 = noisy_clue(ws2, a, epsilon2)

        # compute ws3
        w3 = best_final_guess(w1, c1, w2, c2)
        ws3 = word_lists.answers[w3]

        path = f"{ws1} -> {clues.clue_to_str(c1)} -> {ws2} -> {clues.clue_to_str(c2)} -> {ws3}"
        g = ws3
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
