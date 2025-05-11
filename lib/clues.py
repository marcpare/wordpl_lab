import os
import numpy as np
from math import exp
from lib import word_lists

def clue(guess, answer):
    real_clues = ""
    for (i, letter) in enumerate(guess):
        if letter == answer[i]:
            real_clues += '0' # c
        elif letter in answer:
            real_clues += '1' # i
        else:
            real_clues += '2' # .
    return int(real_clues, 3)

def clue_str_to_int(s):
    t = str.maketrans("ci.GYB", "012012")
    return int(s.translate(t), 3)


def clue_diffs(c1, c2):
    diffs = 0
    for i in range(5):
        if c1 % 3 != c2 % 3:
            diffs += 1
        c1 //= 3
        c2 //= 3
    return diffs

assert clue_diffs(clue_str_to_int("ccccc"), clue_str_to_int("ccccc")) == 0
assert clue_diffs(clue_str_to_int("....."), clue_str_to_int(".....")) == 0
assert clue_diffs(clue_str_to_int("iiiii"), clue_str_to_int("iiiii")) == 0
assert clue_diffs(clue_str_to_int("ccccc"), clue_str_to_int("ciccc")) == 1
assert clue_diffs(clue_str_to_int("ccccc"), clue_str_to_int("c.c.c")) == 2


def clue_to_str(n):
    if n == 0:
        return "ccccc"
    digits = []
    while n:
        digits.append(str(n % 3))
        n //= 3
    s = ''.join(digits[::-1])
    t = str.maketrans("012", "ci.")
    # left pad with cs to make it 5 digits
    return s.zfill(5).translate(t)


def cwa(cache_filepath="data/cwa.txt"):
    """
    The clue for each valid word and answer, stored in a compact base 3 integer

    cwa[guess_word][answer_word]: clue 

    """

    cwa = []
    if not os.path.exists(cache_filepath):
        for gw in word_lists.valid_words:
            row = []
            for aw in word_lists.answers:
                row.append(clue(gw, aw))
            cwa.append(row)
        with open(cache_filepath, "w") as f:
            for row in cwa:
                f.write(" ".join(map(str, row)) + "\n")
    else:
        with open(cache_filepath, "r") as f:
            cwa = [list(map(int, line.split())) for line in f]

    cwa = np.array(cwa)
    return cwa


def d():
    """
    d[clue1][clue2]: count of letters that are different between clue1 and clue2
    """

    dm = []
    for c1 in range(3**5):
        row_d = []
        for c2 in range(3**5):
            dd = sum(1 for a, b in zip(clue_to_str(c1), clue_to_str(c2)) if a != b)
            row_d.append(dd)
        dm.append(row_d)

    assert dm[int("22222", 3)][int("22222", 3)] == 0
    assert dm[int("22222", 3)][int("22221", 3)] == 1
    assert dm[int("01210", 3)][int("01210", 3)] == 0
    assert dm[int("01210", 3)][int("22222", 3)] == 4

    return np.array(dm)

def pd(epsilon):
    """
    pd[clue1][clue2]: the probability of getting a clue given an actual clue
    """
    dm = d()
    pd = []
    # probability of a correct clue
    pc = 1 - 2.0 / (2.0 + exp(epsilon / 5.0))

    # probability of a particular incorrect clue
    pi = 1.0 / (2.0 + exp(epsilon / 5.0))
    n = 5

    for c1 in range(3**5):
        row_pd = []
        for c2 in range(3**5):
            dd = dm[c1][c2]
            k = 5 - dd
            p = pc**k * (pi**(n-k))
            row_pd.append(p)
        pd.append(row_pd)

    return np.array(pd)