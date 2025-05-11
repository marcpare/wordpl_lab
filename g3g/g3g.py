import numpy as np
from lib import word_lists, clues
from tqdm import tqdm


epsilon = 28.0
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

    all_w3 = set()

    for c1 in tqdm(range(3**5)):

        for w2 in highest_entropy_guesses(w1, c1, N=40):

            for c2 in tqdm(range(3**5)):
                
                w3 = best_third_guess(w1, c1, w2, c2)

                if w3 not in all_w3:
                    all_w3.add(w3)
                    print(len(all_w3))