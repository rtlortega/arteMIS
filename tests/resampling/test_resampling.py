import warnings
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from matchms import calculate_scores

from artemis.resampling.resampling import make_subsample_replicates_cosine

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _build_scores(n=6):
    spectra = [
        Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensities=np.array([0.5, 1.0, 0.3]) / (i + 1),
            metadata={"precursor_mz": 300.0 + i, "id": f"s{i}"},
        )
        for i in range(n)
    ]
    return calculate_scores(spectra, spectra, ModifiedCosine())


def test_make_subsample_replicates_cosine_indices_unique_and_aligned():
    scores = _build_scores(6)
    identifiers = [f"s{i}" for i in range(6)]

    replicates = make_subsample_replicates_cosine(
        scores, identifiers, n=3, fraction=0.5
    )

    assert len(replicates) == 3
    for idx, ids in replicates:
        assert len(idx) == len(ids) == len(set(idx))
        assert ids == [identifiers[i] for i in idx]
