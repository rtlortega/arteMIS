import yaml
from matchms.filtering import default_filters, normalize_intensities
from matchms import filtering as msfilters


class SpectrumFilter:
    """
    Preprocess a list of matchms.Spectrum objects based on YAML-defined filters.
    Does NOT read files; input spectra should be passed directly.
    """

    def __init__(self, spectra, yaml_config_path=None):
        """
        Parameters
        ----------
        spectra : list of matchms.Spectrum
            Input spectra to process.
        yaml_config_path : str, optional
            Path to YAML file with filter definitions. If None, default filters are used.
        """
        self.spectra = spectra
        self.yaml_config_path = yaml_config_path
        self.filters = self._load_filters()

    def _load_filters(self):
        """Load filter config from YAML or use defaults."""
        if self.yaml_config_path:
            with open(self.yaml_config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("filters", [])
        else:
            # Default filter chain
            return [
                {"name": "select_by_mz", "params": {"mz_from": 0, "mz_to": 1000}},
                {
                    "name": "select_by_relative_intensity",
                    "params": {"intensity_from": 0.01},
                },
                {"name": "reduce_to_number_of_peaks", "params": {"n_max": 1000}},
                {
                    "name": "require_minimum_number_of_peaks",
                    "params": {"n_required": 5},
                },
                {
                    "name": "require_minimum_number_of_high_peaks",
                    "params": {"no_peaks": 5, "intensity_percent": 2.0},
                },
                {"name": "remove_profiled_spectra", "params": {}},
                {"name": "remove_noise_below_frequent_intensities", "params": {}},
            ]

    def _apply_single_filter(self, spectra, filt):
        """Apply a single filter and print counts."""
        func = getattr(msfilters, filt["name"], None)
        if func is None:
            print(f"⚠️ Filter '{filt['name']}' not found — skipping.")
            return spectra

        filtered = [func(s, **filt.get("params", {})) for s in spectra]
        filtered = [s for s in filtered if s is not None]
        print(f"→ After {filt['name']}: {len(filtered)} spectra remain")
        return filtered

    def process(self):
        """Apply default filters and YAML-defined filters, print summary, and return filtered spectra."""
        spectra = self.spectra

        # Apply default filters first
        spectra = [default_filters(s) for s in spectra]
        spectra = [normalize_intensities(s) for s in spectra]
        spectra = [s for s in spectra if s is not None]
        print(f"→ After default filters: {len(spectra)} spectra remain")

        # Apply YAML or default filters
        for filt in self.filters:
            spectra = self._apply_single_filter(spectra, filt)

        print(f"Final spectra count: {len(spectra)}")
        return spectra
