import numpy as np
from numba import njit, prange
from tqdm import tqdm
from lib import clues, word_lists
import plotly.express as px
import pandas as pd

ws1 = "salet"
w1 = word_lists.valid_words.index(ws1)

c1 = 25
w2 = 2757
c2 = 209
w3 = 708
c3 = 78
w4 = 1977

epsilon = 12.0

prob_dist = clues.pd(epsilon)
cwa = clues.cwa()

# Calculate probabilities for first guess
expected_wins1 = np.zeros(word_lists.NAW)
for a in prange(word_lists.NAW):
    expected_wins1[a] = prob_dist[c1][cwa[w1][a]]
sorted_expected_wins1 = np.sort(expected_wins1)
sorted_expected_wins1 = sorted_expected_wins1 / np.sum(sorted_expected_wins1)

# Calculate probabilities for second guess
expected_wins2 = np.zeros(word_lists.NAW)
for a in prange(word_lists.NAW):
    expected_wins2[a] = prob_dist[c1][cwa[w1][a]] * prob_dist[c2][cwa[w2][a]]
sorted_expected_wins2 = np.sort(expected_wins2)
sorted_expected_wins2 = sorted_expected_wins2 / np.sum(sorted_expected_wins2)

# Calculate probabilities for third guess
expected_wins3 = np.zeros(word_lists.NAW)
for a in prange(word_lists.NAW):
    expected_wins3[a] = prob_dist[c1][cwa[w1][a]] * prob_dist[c2][cwa[w2][a]] * prob_dist[c3][cwa[w3][a]]
sorted_expected_wins3 = np.sort(expected_wins3)
sorted_expected_wins3 = sorted_expected_wins3 / np.sum(sorted_expected_wins3)

# Create DataFrame for plotting
df = pd.DataFrame({
    'Word Index': list(range(word_lists.NAW)) * 3,
    'Probability': np.concatenate([sorted_expected_wins1, sorted_expected_wins2, sorted_expected_wins3]),
    'Guess': ['After first guess'] * word_lists.NAW + ['After second guess'] * word_lists.NAW + ['After third guess'] * word_lists.NAW
})

# Create interactive plot
fig = px.scatter(df, x='Word Index', y='Probability', color='Guess',
                 log_y=True,
                 template='plotly_dark',
                 title='Word Probabilities After Each Guess')

fig.update_layout(
    showlegend=True,
    xaxis_title='Word Index',
    yaxis_title='Probability',
    yaxis=dict(
        tickformat='.1e'  # Use scientific notation with 1 decimal place
    )
)

fig.show()
