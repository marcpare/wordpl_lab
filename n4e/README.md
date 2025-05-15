
g3g

numba: 3.6s / it
multiprocessing numpy: 27s / 6 it (4.5s / it)

numba prange: 0.68 s / it
    also lights up all my CPUs without blowing up memory

numba prange on drop: 0.18 s / it
    25X improvement over parallel numpy
    lights up all cores

***

n4e

compute entropies:

numba prange 1.06 s/it
numpy 7.90 s/it