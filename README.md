WORDPL Solver
=============

This repository implements an exhaustive search solver for WORDPL -- a variant of Wordle that sometimes gives incorrect clues.

[What is WORDPL?](https://github.com/TedTed/wordpl?tab=readme-ov-file)

## Table of Contents

- [Overview](#overview)
- [Algorithm Details](#algorithm-details)
  - [Computing expected wins](#computing-expected-wins)
  - [Pruning second guess with clue entropy](#pruning-second-guess-with-clue-entropy)
  - [Pruning possible answers](#pruning-possible-answers)
- [Putting it all together](#putting-it-all-together)
- [What's Left?](#whats-left)
- [What does this have to tell us about differential privacy?](#what-does-this-have-to-tell-us-about-differential-privacy)

## Overview

The basic idea of exhaustive search is simple: we play WORDPL with every possible sequence of guesses and pick the one that wins the most times.

In traditional Wordle this is computationally feasible because each guess is guaranteed to eliminate a large number of possible answers. This is not the case in WORDPL because it is possible for any clue to be given at any turn, greatly increasing the search space. Without any pruning, the search space for four guesses is too large to exhaust on a typical PC:

```
for every possible first clue:                   243 possible clues
  for every possible second guess:               12,972 valid words to guess from
    for every possible second clue:              243
      for every possible third guess:            12,972
        for every possible third clue:           243
          compute the expected number of wins.   2,315 possible answer words
```

However, it is possible to greatly reduce the search space with the following two pruning steps:

1. Only consider the top 10 second guesses with the [highest clue entropy](https://dominikfay.me/blog/2025/maxclueentropy/) (h/t @DarthPumpkin).
2. Prune possible answers to the top 2% most likely answers after the second guess.

Algorithm Details
=================

Currently, the solver uses a fixed number of guesses (either three or four), trying to maximize the number of games that are won on the final guess.

Computing expected wins
-----------------------

At the core of the solver is a calculation for the number of expected wins after a sequence of guesses.

Say you have a sequence of word guesses (`w1`, `w2`, `w3`) and resulting noisy clues, which may or may not be accurate (`nc1`, `nc2`, `nc3`). What is the probability that a particular answer `a` is the correct answer?

First, you need the probability of being given noisy clue `nc` for an answer `a` with guess `w`. The clues are computed on a character-by-character basis, with a correct clue character being given with probability

$$P(\text{correct clue character}) = 1 - \frac{2}{2 + \exp\left(\frac{\epsilon}{5}\right)}$$

and an incorrect clue character:

$$P(\text{incorrect clue character}) = \frac{2}{2 + \exp\left(\frac{\epsilon}{5}\right)}$$

where $\epsilon$ is the parameter that controls how noisy the clues are.

For this calculation, we are interested in the probability of a _particular_ incorrect clue. Since there are two possibilities to choose from when a clue is wrong, we divide this probability by 2.

$$P(\text{particular incorrect clue character}) = \frac{1}{2 + \exp\left(\frac{\epsilon}{5}\right)}$$

From here, we want to calculate the probability of being given a particular sequence of five clue characters `nc` for a given guess `w` and answer `a`. We start by computing the truthful clue `c` for `a` given `w`. Then, we count the number of characters different between `nc` and `c`, calling it `k`. This count is the number of characters that would have to randomly be incorrect for this clue to be given. The overall probability for noisy clue `nc` then, is:

$$P(\text{nc} \mid w, a) = \left[ P(\text{correct clue character}) \right]^{5 - k} \times \left[ P(\text{particular incorrect clue character}) \right]^{k}$$

This is everything we need to compute the expected number of wins after a sequence of guesses. Here is the actual implementation of computing a best fourth and final guess after an initial sequence of three guesses:

```
for a in range(number_answer_words):
    expected_wins[a] = pd[nc1][cwa[w1][a]] * pd[nc2][cwa[w2][a]] * pd[nc3][cwa[w3][a]]
```

Breaking it down: the implementation precomputes the probabilities and clues to avoid repeated computations--

`pd[nc][c]` is the probability of being given a noisy clue `nc` if the actual clue is `c`

`cwa[word][answer]` is the clue given for `answer` if `word` is guessed

So, `pd[nc1][cwa[w1][a]]` is $$P(nc | w , a)$$, the probability of being given noisy clue `nc1` if word `w1` is guessed and the actual answer is `a`.

For a sequence of multiple guesses, the probabilities multiply together into the final probability. By running this over all possible answers (`for a in answers`), we can find the answer which is the most likely to be correct as well as the fraction of all possible WORDPL games in which this guess wins the game.


Pruning second guess with clue entropy
--------------------------------------

The number of words considered for a second guess can be greatly reduced using a maximum entropy approach.

```
for every possible first clue:                   243 possible clues
  for every possible second guess:           --> 12,972 valid words to guess from
    for every possible second clue:              243
      for every possible third guess:            12,972
        for every possible third clue:           243
          compute the expected number of wins.   2,315 possible answer words
```

Instead of searching all 12,972 words, we want to limit the set of words to search to those that are likely to win the game the most times.

To do this, the clue entropy is computed for each possible second word guess and the top 10 are used as candidates for further searching. 

The ins and outs of the maximum entropy approach are explained by Dominik Fay at https://dominikfay.me/blog/2025/maxclueentropy/

Why not just use the word that maximizes clue entropy? Well, like in traditional Wordle, sometimes a guess early on in the game makes it harder to win the game in future turns. A guess that maximizes clue entropy may give you the most information about the true answer after turn 2, but it may leave you without a good choice of word for a third guess to narrow the space even further, since you can't choose arbitrary sequences of characters, only the subset that are valid guesses. Here is an example comparison of clue entropy vs. expected wins for a sample second guess:

![Screenshot 2025-05-22 at 9 10 28 AM](https://github.com/user-attachments/assets/16f1f4de-b48d-447e-8616-a79f5ac3ae57)

The highest entropy guess is marked with a green circle; it is much worse than the best possible guess!

Pruning possible answers
------------------------

After making three guesses, it should not be necessary to consider every possible answer word `a` as a solution because some will be much more likely than others given the clues provided so far.

```
for every possible first clue:                    243 possible clues
  for every possible second guess:                12,972 valid words to guess from
    for every possible second clue:               243
      for every possible third guess:             12,972
        for every possible third clue:            243
          compute the expected number of wins --> 2,315 possible answer words
```

We prune the search over possible answers for an optimal final guess by computing the likelihood of each possible answer given the first two clues and then use only the top 2% of these possible answers for the subsequent evaluation for the optimal third guess. Or, in other words, we only evaluate the best possible third guess word against the most likely 2% of answers that it could be after two guesses.

The figures below show the changing likelihoods for all answers as guesses are made for a game with ε=12:

![Screenshot 2025-05-22 at 9 17 27 AM](https://github.com/user-attachments/assets/fcd80ee4-40d5-48cd-940f-b1c5b489b586)


Note that the y-axis is on a log scale -- the space of possible answers converges quickly, even with the significant amount of noise added at ε=12.

Zooming to the most likely answers, we see the most likely possible answers is 10 times more likely than the next 25 possible answers, which themselves are 10 times more likely than the next most likely answers.

![Screenshot 2025-05-22 at 9 20 38 AM](https://github.com/user-attachments/assets/95e9aac1-1f81-4cf1-aecf-e8ec844d0663)


Putting it all together
=======================

The solver described here is implemented in about a page of code:

https://github.com/marcpare/wordpl_lab/blob/main/n4e/n4e.py

With another page of library code for precomputing the `pd` and `cwa` arrays:

https://github.com/marcpare/wordpl_lab/blob/main/lib/clues.py

For the 95th and 50th percentile games, a four-guess strategy is used starting with the [optimal starting word from Wordle](https://sonorouschocolate.com/notes/index.php?title=The_best_strategies_for_Wordle) of `salet`.

The 5th percentile game differs, employing a three-guess strategy. In traditional Wordle, about 50% of games can be won by three guesses. This percentage is too low for the 50th and 95th percentile versions of WORDPL, but it is more than high enough for the 5th percentile version. The additional accuracy gained from a fourth guess is offset by a reduced clue accuracy from smaller $\epsilon$.

Because three-guess strategies are so quick to compute, it was possible to explore allocating ε differently across guesses. Interestingly this does make a difference with about 1% more wins achieved by allocating a high ε to the first guess of a three-guess strategy. The optimal distribution was found to be about 2:1.


What's Left?
=======================

Exhaustive searches are provably optimal (they try everything!), but this work hasn't been truly exhaustive in its search. The pruning parameters (top 10 highest entropy second guesses and top 2% possible answers) might have been too aggressive. Indeed, the example figure for entropy vs score appears to show an optimal guess outside the top 10 highest entropy candidates.

This work also did not explore the optimal ε distribution across guesses. The preliminary investigation of ε distribution for the three-guess suggests there may be accuracy to be gained here.

Finally, only one starting word was studied for the four-guess strategy. It is possible that the best starting word for WORDPL is not the same as the best starting word for Wordle.

And what about something dramatically different? Could a strategy with many low ε guesses beat the three and four-guess strategies? Maybe!


What does this have to tell us about differential privacy?
----------------------------------------------------------

WORDPL was built as a demonstration of differential privacy by the team at [Oblivious](https://www.oblivious.com/)

> Each game lets you adjust the epsilon parameter, allowing you to balance the trade-off between privacy and accuracy in an interactive environment.

In its original form, the game was played once through, ranking results by ε required for a win. Damien (@TedTed) reframed this one-shot version of the game into one that more closely mimics real-life deployment of differential privacy by scoring results based on performance over many trials.

This reframing does a good job to highlight one of the non-intuitive aspects of statistical privacy methods: data that feel noisy to human beings can still represent significant privacy risk. Consider the best strategy for winning WORDPL at least 5% of the time. This strategy requires ε = 15.6, which almost always provides an incorrect clue at each round, and often more than one. To a human being, this makes the game incredibly difficult. However, with optimal strategy, it is possible to reliably win 1 out of 20 times. 

Imagine if we were instead talking about the US Census. And instead of guessing a word, we were guessing a person's name. Would regulators be happy that 1 out of 20 people could be identified reliably? Of the USA's 370 million people, that would affect 17 million. That's nearly the population of New York state!

Of course, here we are talking about a toy example to demonstrate one aspect of statistical privacy methods (the trade-off between privacy and accuracy). In practice, algorithm design (basically, "asking the right question") allows useful outputs for data analysts at much lower ε values. Maybe one day there will be a game developed to illustrate this!