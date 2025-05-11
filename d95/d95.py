import math
import numpy as np
from lib import word_lists, clues

epsilon = 30.0
w1 = word_lists.valid_words.index("salet")

cwa = clues.cwa()
d = clues.d()
pd = clues.pd(epsilon)


def prune_possible_answers(answers, w, c):
    # prune based on all clues
    # for each solution, count the number of diffs with each guess/clue pair
    # return all solutions with the minimum

    min_diffs = float('inf')
    min_diffs_solutions = []

    for answer in answers:
        total_diffs = d[cwa[w][answer]][c]
        if total_diffs < min_diffs:
            min_diffs = total_diffs
            min_diffs_solutions = [answer]
        elif total_diffs == min_diffs:
            min_diffs_solutions.append(answer)

    return min_diffs_solutions, min_diffs


def expected_last_turn_wins(game_state):

    ps = None

    for i in range(len(game_state) // 2):
        w = game_state[2*i]
        c = game_state[2*i+1]

        if ps is None:
            ps = pd[c][cwa[w]]
        else:
            ps = ps * pd[c][cwa[w]]

    ps = ps[:, np.newaxis]

    return np.max(ps), np.argmax(ps)


def num_wins_from(game_state, possible_answers):

    turn = len(game_state) // 2

    max_wins = 0
    best_guess = None

    if len(possible_answers) == 1:
        nw, bg = expected_last_turn_wins(game_state)                
        return nw, word_lists.valid_words.index(word_lists.answers[bg])

    guesses_by_entropy = []
    for guess in range(word_lists.NVW):
        # partition possible answers by clue
        answer_by_clue = [[] for _ in range(3**5)]

        for answer in possible_answers:
            clue = cwa[guess][answer]
            answer_by_clue[clue].append(answer)

        # compute entropy of this guess
        entropy = 0
        for c in range(3**5):
            if len(answer_by_clue[c]) == 0:
                continue
            p = len(answer_by_clue[c]) / len(possible_answers)
            entropy -= p * math.log(p)

        guesses_by_entropy.append((guess, entropy))

    # sort by entropy decreasing
    guesses_by_entropy.sort(key=lambda x: x[1], reverse=True)

    for w, _ in guesses_by_entropy[:40]:

        num_wins = 0

        # partition possible answers by clue
        answer_by_clue = [[] for _ in range(3**5)]
        for answer in possible_answers:
            clue = cwa[w][answer]
            answer_by_clue[clue].append(answer)

        for c in range(3**5):

            if len(answer_by_clue[c]) == 0:
                continue

            gs = game_state + [w, c]
            pas = answer_by_clue[c]

            if len(pas) == 1:
                nw, bg = expected_last_turn_wins(gs)                
                bg = word_lists.valid_words.index(word_lists.answers[bg])
            elif turn == 3:
                nw, bg = expected_last_turn_wins(gs)
                bg = word_lists.valid_words.index(word_lists.answers[bg])
            else:
                nw, bg = num_wins_from(gs, pas)

            num_wins += nw

        if num_wins > max_wins:
            max_wins = num_wins
            best_guess = w

    return max_wins, best_guess


if __name__ == "__main__":

    strategy = []

    for c1 in range(3**5):
        pas, min_diffs = prune_possible_answers(range(word_lists.NAW), w1, c1)
        max_wins, best_guess = num_wins_from([w1, c1], pas)
        print(clues.clue_to_str(c1), word_lists.valid_words[best_guess], min_diffs, len(pas), max_wins)

        strategy.append(c1)

    print(strategy)