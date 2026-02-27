"""
Prob and Stats Lab – Discrete Probability Distributions
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    """
    A = first card is an Ace
    B = second card is an Ace
    """

    rng = np.random.default_rng(42)

    # Theoretical probabilities
    P_A = 4 / 52
    P_B = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = (4 / 52) * (3 / 51)

    # Simulation
    trials = 200_000
    deck = np.array([1]*4 + [0]*48)

    A_count = 0
    AB_count = 0

    for _ in range(trials):
        draw = rng.choice(deck, size=2, replace=False)
        if draw[0] == 1:
            A_count += 1
            if draw[1] == 1:
                AB_count += 1

    empirical_P_A = A_count / trials
    empirical_P_B_given_A = AB_count / A_count

    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):

    rng = np.random.default_rng(42)

    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    samples = rng.binomial(1, p, 100_000)
    empirical_P_X_1 = np.mean(samples)

    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):

    rng = np.random.default_rng(42)

    def comb(n, k):
        return math.comb(n, k)

    theoretical_P_0 = (1 - p) ** n
    theoretical_P_2 = comb(n, 2) * (p**2) * ((1 - p) ** (n - 2))
    theoretical_P_ge_1 = 1 - theoretical_P_0

    samples = rng.binomial(n, p, 100_000)
    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error = abs(theoretical_P_ge_1 - empirical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():

    rng = np.random.default_rng(42)
    p = 1 / 6

    theoretical_P_1 = p
    theoretical_P_3 = ((5/6)**2) * p
    theoretical_P_gt_4 = (5/6)**4

    samples = rng.geometric(p, 200_000)
    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(theoretical_P_gt_4 - empirical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):

    rng = np.random.default_rng(42)

    def poisson_pmf(k, lam):
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    theoretical_P_0 = poisson_pmf(0, lam)
    theoretical_P_15 = poisson_pmf(15, lam)

    theoretical_P_ge_18 = 1 - sum(poisson_pmf(k, lam) for k in range(18))

    samples = rng.poisson(lam, 100_000)
    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(theoretical_P_ge_18 - empirical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )