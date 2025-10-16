import warnings
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from matchms import calculate_scores

from scr.bootstrapping.bootstrapping import boostrap_spectra_replacement

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
