import numpy as np
import json
from lib import word_lists, clues
from tqdm import tqdm


epsilon = 12.0
ws1 = "salet"

print("Assessing epsilon =", epsilon)
w1 = word_lists.valid_words.index(ws1)
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

    for w3 in range(word_lists.NVW):

        ps3 = pd[:][cwa[w3]]
        ps4 = ps * ps3
        expected_wins = np.sum(np.max(ps4, axis=0))

        if expected_wins >= best_expected_wins:
            best_expected_wins = expected_wins
            best_w3 = w3

    third_guess_memo[memo_key] = best_w3
    return best_w3


def highest_entropy_guesses(w1, c1, N=40):
    word_entropies = []

    psp = pd[c1][cwa[w1]]
    psp = psp[:, np.newaxis]

    for w in range(word_lists.NVW):
        ps = np.sum(psp * pd[:][cwa[w]], axis=0)
        ps = ps / np.sum(ps) # normalize
        entropy = -np.sum(ps * np.log2(ps))
        word_entropies.append((w, entropy))

    word_entropies.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in word_entropies[:N]]



if __name__ == "__main__":

    # Evaluate heuristics for third guess

    with open("d95/d95_third_guess_memo.json") as f:
        third_guess_memo = json.load(f)

    for path, bw3 in third_guess_memo.items():
        w1, c1, w2, c2 = path.split(" ")

        w1 = int(w1)
        c1 = int(c1)
        w2 = int(w2)
        c2 = int(c2)

        ps1 = pd[c1][cwa[w1]]
        ps2 = pd[c2][cwa[w2]]
        ps = ps2[:, np.newaxis]

        # answers that make up 90% of outcomes
        ps = np.sort(ps)
        sps = np.cumsum(ps)

        sps = sps / sps[-1]

        # plot sps, one bar per value
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.bar(range(len(sps)), sps)
        plt.xlabel('Answer Index')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Answer Probabilities')
        plt.grid(True)
        plt.show()


