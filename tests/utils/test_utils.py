import warnings

from rdkit import DataStructs
from lattice.utils.fps import smiles_to_morgan_fps

warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_smiles_to_morgan_fps():
    # Test valid SMILES
    smiles = "CCO"
    fps = smiles_to_morgan_fps(smiles)
    assert fps is not None
    assert isinstance(fps, DataStructs.cDataStructs.ExplicitBitVect)

    # Test invalid SMILES
    invalid_smiles = "invalid_smiles"
    fps_invalid = smiles_to_morgan_fps(invalid_smiles)
    assert fps_invalid is None

    # Test empty string
    empty_input = ""
    fps_empty = smiles_to_morgan_fps(empty_input)
    assert fps_empty is not None  # empty string returns None fingerprint

    # Test NaN input
    nan_input = float("nan")
    fps_nan = smiles_to_morgan_fps(nan_input)
    assert fps_nan is None
