
valid_words = open("data/valid.txt", "r").read().splitlines()
answers = open("data/answers.txt", "r").read().splitlines()
good_words = open("data/good.txt", "r").read().splitlines()

good_word_indices = [valid_words.index(gw) for gw in good_words]

NAW = len(answers)
NVW = len(valid_words)
VW = range(NVW)
AW = range(NAW)
