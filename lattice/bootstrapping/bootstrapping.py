from sparsestack import StackedSparseArray
from matchms import Scores
import numpy as np
from scipy.stats import qmc


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


def get_latin_hypercube_samples(settings: dict, num_samples: int, seed: int):
    """Generate parameter sets using Latin Hypercube Sampling.
    Parameters
    ----------
    settings: dict
        Dictionary with parameter names as keys and (min, max) tuples as values.
    num_samples: int
        Number of parameter sets to generate.
    seed: int
        Random seed for reproducibility.
    Returns
    -------
    param_sets: list of dict
        List of dictionaries with parameter sets.
    unit_samples: np.ndarray
        Array of samples in the unit hypercube [0,1].
    """
    sampler = qmc.LatinHypercube(d=len(settings), optimization="random-cd", seed=seed)
    unit_samples = sampler.random(num_samples)  # always in [0,1]

    l_bounds = [bound[0] for bound in settings.values()]
    u_bounds = [bound[1] for bound in settings.values()]

    scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)

    param_names = list(settings.keys())
    param_sets = []
    for row in scaled_samples:
        params = {}
        for name, val in zip(param_names, row):
            params[name] = int(round(val)) if name != "cut_off" else round(val, 2)
        param_sets.append(params)

    return param_sets, unit_samples
