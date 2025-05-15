import random
import math
import numpy as np
from scipy.stats import beta
from lib import word_lists, clues
from numba import njit, prange

# trace [12.35, 12.35, 12.35]
# (2317, 0.5071212775140268, (0.48676482598897136, 0.5274660844838033))

# trace [13.35, 13.35, 10.35]
# (2317, 0.5049633146309883, (0.48460756551183404, 0.5253109478478754))

# trace [27.0, 27.0, 27.0]
# (2317, 0.9546827794561934, (0.9458506919434044, 0.9627718226775013))

# salet [12.35, 12.35, 12.35]
# (2317, 0.5235217954251187, (0.5031724105849317, 0.543832717955989))

# salet [12.3, 12.3, 12.3]
# (2317, 0.5079844626672422, (0.4876278363926947, 0.5283280329253488))

epsilon1 = 12.3
epsilon2 = 12.3
epsilon3 = 12.3


ws1 = "salet"

strategy = []
with open("numba_test/n4e_salet.txt", "r") as f:
    for line in f:
        idx, w2 = line.split()
        strategy.append(int(w2))

print("Assessing epsilon1 =", epsilon1, "epsilon2 =", epsilon2, "epsilon3 =", epsilon3)
w1 = word_lists.valid_words.index(ws1)

pd1 = clues.pd(epsilon1)
pd2 = clues.pd(epsilon2)
pd3 = clues.pd(epsilon3)

cwa = clues.cwa()

#############################################################################################################
#
# Search for best third guess in a four guess strategy
#
#############################################################################################################

@njit(parallel=True)
def best_third_guess(w1, c1, w2, c2):

    # likelihood of answer a given w1, c1, w2, c2
    lia = np.zeros(word_lists.NAW)
    for a in range(word_lists.NAW):
        lia[a] = pd1[c1][cwa[w1][a]] * pd2[c2][cwa[w2][a]]
    slia = np.argsort(lia)
    possible_answers = slia[-int(word_lists.NAW / 50):]

    expected_wins = np.zeros(word_lists.NVW)

    for w3 in prange(word_lists.NVW):
        total_expected_wins = 0
        for c3 in range(3**5):
            bew = 0
            for a in possible_answers:
                bew = max(bew, pd1[c1][cwa[w1][a]] * pd2[c2][cwa[w2][a]] * pd3[c3][cwa[w3][a]])
            total_expected_wins += bew
        expected_wins[w3] = total_expected_wins

    return np.argmax(expected_wins)


@njit(parallel=True)
def best_final_guess(w1, c1, w2, c2, w3, c3):
    expected_wins = np.zeros(word_lists.NAW)
    for a in prange(word_lists.NAW):
        expected_wins[a] = pd1[c1][cwa[w1][a]] * pd2[c2][cwa[w2][a]] * pd3[c3][cwa[w3][a]]
    return np.argmax(expected_wins)


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

    shuffled_answers = random.sample(word_lists.answers, len(word_lists.answers))

    for a in shuffled_answers:
        c1 = noisy_clue(ws1, a, epsilon1)
        w2 = strategy[c1]
        ws2 = word_lists.valid_words[w2]
        c2 = noisy_clue(ws2, a, epsilon2)

        # compute ws3
        w3 = best_third_guess(w1, c1, w2, c2)
        ws3 = word_lists.valid_words[w3]
        c3 = noisy_clue(ws3, a, epsilon3)

        w4 = best_final_guess(w1, c1, w2, c2, w3, c3)
        ws4 = word_lists.answers[w4]

        path = f"{ws1} -> {clues.clue_to_str(c1)} -> {ws2} -> {clues.clue_to_str(c2)} -> {ws3} -> {clues.clue_to_str(c3)} -> {ws4}"

        g = ws4

        if g == a:
            estimator.update(1)
        else:
            estimator.update(0)

        print(path, "WIN" if g == a else "LOSE", estimator.get_estimate())