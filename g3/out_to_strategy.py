
from lib import clues, word_lists

with open("g3/g3_out.txt") as f:

    strategy = {}

    for line in f:
        sp = line.split(" ")
        c = clues.clue_str_to_int(sp[0])
        w = word_lists.valid_words.index(sp[2])

        if c in strategy:
            assert w == strategy[c]
        
        strategy[c] = w

    for c in range(3**5):
        if c not in strategy:
            print("missing", clues.clue_to_str(c))
        
    strategy = [strategy[c] for c in range(3**5)]
    print(strategy)