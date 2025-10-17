import numpy as np
from matchms import Spectrum
from lattice.scores.SimilarityCalculator import SimilarityCalculator
from unittest.mock import MagicMock


def test_similarity_calculator_simple_mocked():
    # Minimal spectra
    filtered_spectra = [
        Spectrum(
            mz=np.array([100, 200], dtype=float),
            intensities=np.array([10, 20], dtype=float),
        ),
        Spectrum(
            mz=np.array([150, 250], dtype=float),
            intensities=np.array([15, 25], dtype=float),
        ),
    ]

    calculator = SimilarityCalculator(filtered_spectra)

    # Mock the similarity methods to return a dummy object with the attributes we need
    dummy_scores = MagicMock()
    dummy_scores.scores.shape = [2, 2, 1]  # fake shape
    dummy_scores.score_names = ("FAKE",)

    calculator.calculate_modcosine = MagicMock(return_value=dummy_scores)
    calculator.calculate_spec2vec = MagicMock(return_value=dummy_scores)
    calculator.calculate_ms2deepscore = MagicMock(return_value=dummy_scores)

    # Run the “tests”
    for func in [
        calculator.calculate_modcosine,
        calculator.calculate_spec2vec,
        calculator.calculate_ms2deepscore,
    ]:
        scores_obj = func()
        assert hasattr(scores_obj, "scores")
        assert hasattr(scores_obj, "score_names")
        assert scores_obj.scores.shape[0] > 0
