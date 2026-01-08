from scipy.stats import qmc


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
