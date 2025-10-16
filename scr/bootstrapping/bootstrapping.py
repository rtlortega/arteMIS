from sparsestack import StackedSparseArray
from matchms import Scores
import numpy as np


def boostrap_spectra_replacement(scores: Scores, seed: int) -> Scores:
    """Create a bootstrap sample with replacement from a Scores object matchms.

    Parameters
    ----------
    scores: matchms.Scores
        Scores object to create the bootstrap sample from.
    seed: int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    indices = np.random.choice(
        len(scores.references), len(scores.references), replace=True
    )
    new_refs = scores.references[indices]
    new_queries = scores.queries[indices]
    index_map = {old: new for new, old in enumerate(indices)}

    # here it obtains the column and ro and data for the sparse score matrix
    mask = np.isin(scores._scores.row, indices) & np.isin(scores._scores.col, indices)

    old_rows = scores._scores.row[mask]
    old_cols = scores._scores.col[mask]
    old_data = scores._scores.data[mask]

    new_rows = np.array([index_map[r] for r in old_rows])
    new_cols = np.array([index_map[c] for c in old_cols])
    new_data = old_data

    new_stack = StackedSparseArray(len(new_refs), len(new_queries))
    new_stack.add_sparse_data(new_rows, new_cols, new_data, name="")

    new_scores = Scores(new_refs, new_queries)
    new_scores._scores = new_stack
    return new_scores
