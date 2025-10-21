import warnings
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from matchms import calculate_scores

from lattice.bootstrapping.bootstrapping import boostrap_spectra_replacement

warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_bootstrap_spectra_replacement():
    # Create dummy spectra
    spectra1 = [
        Spectrum(
            mz=np.array([100, 150, 200], dtype=float),
            intensities=np.array([10, 20, 30], dtype=float),
            metadata={"id": f"spec_{i}", "precursor_mz": 200.0},
        )
        for i in range(5)
    ]
    spectra2 = [
        Spectrum(
            mz=np.array([100, 150, 200], dtype=float),
            intensities=np.array([15, 25, 35], dtype=float),
            metadata={"id": f"spec_{i}", "precursor_mz": 150.0},
        )
        for i in range(5)
    ]

    # Calculate initial scores
    similarity = ModifiedCosine(tolerance=0.1)
    scores = calculate_scores(spectra1, spectra2, similarity)

    # Perform bootstrapping
    bootstrapped_scores = boostrap_spectra_replacement(scores, seed=42)

    # Check that the bootstrapped scores have the same number of references and queries
    assert len(bootstrapped_scores.references) == len(scores.references)
    assert len(bootstrapped_scores.queries) == len(scores.queries)

    # Check that the bootstrapped scores contain valid indices
    for ref in bootstrapped_scores.references:
        assert ref in scores.references
    for query in bootstrapped_scores.queries:
        assert query in scores.queries


def test_get_latin_hypercube_samples():
    from lattice.bootstrapping.bootstrapping import get_latin_hypercube_samples

    # --- Setup ---
    settings = {
        "param1": (0, 10),
        "param2": (5, 15),
        "cut_off": (0.0, 1.0),
    }

    num_samples = 50
    seed = 42

    # --- Run function ---
    param_sets, unit_samples = get_latin_hypercube_samples(settings, num_samples, seed)

    # --- Basic checks ---
    assert isinstance(param_sets, list)
    assert isinstance(unit_samples, np.ndarray)
    assert len(param_sets) == num_samples
    assert unit_samples.shape == (num_samples, len(settings))

    # --- Non-empty and proper bounds ---
    assert num_samples > 0
    assert np.all(unit_samples >= 0) and np.all(unit_samples <= 1)

    # --- Each param_set must have all keys ---
    for params in param_sets:
        assert set(params.keys()) == set(settings.keys())

    # --- Check scaling is within bounds ---
    for params in param_sets:
        for key, (low, high) in settings.items():
            assert low <= params[key] <= high, f"{key} out of bounds: {params[key]}"

    # --- Check reproducibility (same seed → same result) ---
    param_sets2, unit_samples2 = get_latin_hypercube_samples(
        settings, num_samples, seed
    )
    np.testing.assert_allclose(unit_samples, unit_samples2)
    assert param_sets == param_sets2

    # --- Check randomness (different seed → different result) ---
    _, unit_samples_diff = get_latin_hypercube_samples(settings, num_samples, seed + 1)
    assert not np.allclose(unit_samples, unit_samples_diff)
