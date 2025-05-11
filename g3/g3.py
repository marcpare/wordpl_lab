import numpy as np
from tqdm import tqdm
from multiprocessing import Queue, Process
import multiprocessing as mp

from lib import clues, word_lists

ws1 = "trace"
epsilon1 = 9.3
epsilon2 = 5.3

pd1 = clues.pd(epsilon1)
pd2 = clues.pd(epsilon2)

cwa = clues.cwa()

def best_second_guess(ws1, c1):

    w1 = word_lists.valid_words.index(ws1)

    ps = pd1[c1][cwa[w1]]
    ps = ps[:, np.newaxis]

    best_w2 = None
    best_expected_wins = 0

    for w2 in range(word_lists.NVW):

        ps2 = pd2[:][cwa[w2]]
        ps3 = ps * ps2
        expected_wins = np.sum(np.max(ps3, axis=0))

        if expected_wins >= best_expected_wins:
            best_expected_wins = expected_wins
            best_w2 = w2

    return word_lists.valid_words[best_w2], best_expected_wins


def worker_fn(clue_queue, results_queue):
    while True:
        c1 = clue_queue.get()
        ws2, expected_wins = best_second_guess(ws1, c1)
        results_queue.put((c1, ws2, expected_wins))

if __name__ == "__main__":

    results_queue = Queue()
    clue_queue = Queue()

    workers = []

    for c1 in range(3**5):
        clue_queue.put(c1)

    for _ in range(int(mp.cpu_count() / 1.5)):
        worker = Process(target=worker_fn, args=(clue_queue, results_queue))
        worker.start()
        workers.append(worker)

    total_expected_wins = 0
    finished_tasks = 0
    while finished_tasks < 3**5:
        c1, ws2, expected_wins = results_queue.get()
        total_expected_wins += expected_wins
        finished_tasks += 1
        print(f"{clues.clue_to_str(c1)} -> {ws2} (expected wins: {expected_wins})", total_expected_wins)

    print("done")

# 2315

# salet 7.3 --> 112.53
# trace 7.3 --> 113.62
# crate 7.3 --> 112.57

# salet 8.3, 6.3 --> 117.63 (better!)
# salet 8.8, 5.8 --> 118.60
# salet 9.3, 5.3 --> 118.73 (best)
# salet 10.3, 4.3 --> 118.48

# salet 6.3, 8.3 --> 116.87
# salet 4.3, 10.3 --> 116.53

# trace 9.3, 5.3 --> 120.20 // 5.19%
# trace 9.2, 5.2 --> 115.5

# soare 9.3, 5.3 --> 114.83