"""
Group comparisons of window-level estimates.

Every comparison in the paper is a difference of group medians tested by
permutation. The test is two-sided on the absolute difference. When the
number of distinct label assignments is at most `draws` it is enumerated and
the p-value is exact; otherwise labels are drawn at random and
p = (hits + 1) / (draws + 1), whose Monte Carlo standard error is reported.

Each call builds its own generator from `seed`, so a p-value does not depend
on which other tests ran before it.
"""

from itertools import combinations
from math import comb

import numpy as np

DRAWS = 20_000
SEED = 20260915


def median_gap(a, b) -> float:
    """median(b) - median(a)."""
    return float(np.median(np.asarray(b, float)) - np.median(np.asarray(a, float)))


def permutation_test(a, b, draws: int = DRAWS, seed: int = SEED) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("both groups need at least one value")
    pool = np.concatenate([a, b])
    n_a, n = len(a), len(pool)
    observed = median_gap(a, b)
    target = abs(observed) - 1e-12
    assignments = comb(n, n_a)

    if assignments <= draws:
        hits = 0
        index = np.arange(n)
        for chosen in combinations(range(n), n_a):
            mask = np.zeros(n, bool)
            mask[list(chosen)] = True
            hits += abs(np.median(pool[~mask]) - np.median(pool[mask])) >= target
        p = hits / assignments
        return {"gap": observed, "p": float(p), "exact": True,
                "draws": assignments, "mc_se": 0.0, "n_a": n_a, "n_b": len(b)}

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        perm = rng.permutation(pool)
        hits += abs(np.median(perm[n_a:]) - np.median(perm[:n_a])) >= target
    p = (hits + 1) / (draws + 1)
    return {"gap": observed, "p": float(p), "exact": False, "draws": draws,
            "mc_se": float(np.sqrt(p * (1 - p) / draws)), "n_a": n_a, "n_b": len(b)}
