import numpy as np


def make_subsample_replicates_cosine(
    scores, identifiers: list, n: int, fraction: float, seed0: int = 0
) -> list:
    """
    Return list of `n` (idx_array, selected_identifiers) tuples.

    Only the sampled index arrays are stored — no score matrices — so memory
    usage is O(n * k) where k is the subsample size, not O(n * k^2).
    The caller is responsible for slicing the score matrix on-the-fly.
    """
    n_total = scores.shape[0]
    n_samples = max(2, int(np.ceil(fraction * n_total)))
    result = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        idx = np.sort(rng.choice(n_total, size=n_samples, replace=False))
        result.append((idx, [identifiers[j] for j in idx]))
    return result
