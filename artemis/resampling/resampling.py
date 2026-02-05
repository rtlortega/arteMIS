from sparsestack import StackedSparseArray
from matchms import Scores
import numpy as np


def subsample_spectra_no_replacement(
    scores: Scores,
    seed: int,
    fraction: float = 0.85,
    n_samples: int | None = None,
) -> Scores:
    """Create a subsample (WITHOUT replacement) from a matchms Scores object.

    Parameters
    ----------
    scores: matchms.Scores
        Scores object to subsample from.
    seed: int
        Random seed for reproducibility.
    fraction: float
        Fraction of spectra to keep (ignored if n_samples is set).
    n_samples: int or None
        Exact number of spectra to keep. If None, uses fraction * N.

    Returns
    -------
    matchms.Scores
        New Scores object containing only the selected spectra and the induced
        sparse score matrix restricted to them.
    """
    rng = np.random.default_rng(seed)
    n_total = len(scores.references)

    if n_samples is None:
        if not (0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_samples = int(np.ceil(fraction * n_total))

    # guardrails: network needs at least 2 nodes to have edges
    n_samples = max(2, min(n_samples, n_total))

    # sample UNIQUE indices (no replacement)
    indices = rng.choice(n_total, size=n_samples, replace=False)

    # optional: keep original order (helps stability of identifiers / debugging)
    indices = np.sort(indices)

    new_refs = scores.references[indices]
    new_queries = scores.queries[indices]

    # map old -> new indices for remapping sparse matrix
    index_map = {old_idx: new_pos for new_pos, old_idx in enumerate(indices)}

    # restrict sparse matrix to the selected indices (induced submatrix)
    mask = np.isin(scores._scores.row, indices) & np.isin(scores._scores.col, indices)

    old_rows = scores._scores.row[mask]
    old_cols = scores._scores.col[mask]
    old_data = scores._scores.data[mask]

    new_rows = np.fromiter(
        (index_map[r] for r in old_rows), dtype=int, count=len(old_rows)
    )
    new_cols = np.fromiter(
        (index_map[c] for c in old_cols), dtype=int, count=len(old_cols)
    )
    new_data = old_data

    new_stack = StackedSparseArray(len(new_refs), len(new_queries))
    new_stack.add_sparse_data(new_rows, new_cols, new_data, name="")

    new_scores = Scores(new_refs, new_queries)
    new_scores._scores = new_stack
    return new_scores


def make_subsample_replicates(scores, n, fraction, seed0=0):
    """
    Return list of `n` subsampled Scores objects using subsample_spectra_no_replacement.
    """
    return [
        subsample_spectra_no_replacement(scores, seed=seed0 + i, fraction=fraction)
        for i in range(n)
    ]
