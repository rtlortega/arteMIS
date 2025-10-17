import pytest
import numpy as np
import yaml
import tempfile
from matchms import Spectrum
from lattice.preprocessing.SpectrumFilter import SpectrumFilter


@pytest.fixture
def dummy_spectra():
    """Return spectra with enough peaks to survive default filters."""
    return [
        Spectrum(
            mz=np.array([100, 150, 200, 250, 300], dtype=float),  # 5 peaks
            intensities=np.array([10, 20, 50, 15, 25], dtype=float),
            metadata={"id": "spec1"},
        ),
        Spectrum(
            mz=np.array([110, 160, 210, 260, 310], dtype=float),  # 5 peaks
            intensities=np.array([30, 10, 40, 20, 15], dtype=float),
            metadata={"id": "spec2"},
        ),
    ]


@pytest.fixture
def yaml_file():
    """Return path to a temporary YAML file with one simple filter."""
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    yaml.dump(
        {
            "filters": [
                {"name": "select_by_mz", "params": {"mz_from": 50, "mz_to": 400}}
            ]
        },
        tmp,
    )
    tmp.close()
    yield tmp.name
    import os

    os.remove(tmp.name)


def test_default_filters(dummy_spectra):
    """Test processing with default filters."""
    processor = SpectrumFilter(spectra=dummy_spectra)
    filtered = processor.process()

    assert isinstance(filtered, list)
    assert len(filtered) > 0
    assert all(hasattr(s, "peaks") for s in filtered)


def test_yaml_filters(dummy_spectra, yaml_file):
    """Test processing with YAML-defined filters."""
    processor = SpectrumFilter(spectra=dummy_spectra, yaml_config_path=yaml_file)
    filtered = processor.process()

    assert isinstance(filtered, list)
    assert len(filtered) > 0
    assert all(hasattr(s, "peaks") for s in filtered)


def test_missing_filter(dummy_spectra):
    """Test that an invalid filter name is skipped gracefully."""
    bad_yaml = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    yaml.dump({"filters": [{"name": "non_existent_filter", "params": {}}]}, bad_yaml)
    bad_yaml.close()

    processor = SpectrumFilter(spectra=dummy_spectra, yaml_config_path=bad_yaml.name)
    filtered = processor.process()

    assert isinstance(filtered, list)
    assert len(filtered) > 0  # Should still return spectra
    import os

    os.remove(bad_yaml.name)
