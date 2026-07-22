<p align="center">
  <img src="docs/artemis_logo.svg" alt="arteMIS logo" width="220"/>
</p>

# arteMIS

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.13%2B-informational.svg)](#installation)

arteMIS (**A**ccelerated **R**anking and **T**uning using multi-metric **I**nterpretability across **S**cores) is a framework that builds optimal molecular networks for downstream analysis. It can start from:

- **global-mode** — in-silico annotated metabolic features (MS2Query-based),
- **seed-mode** — a subset of well-annotated metabolic features, or
- **target-mode** — network construction optimised for a specific class of compounds.

---

## Quick start

### Installation

```bash
git clone https://github.com/rtlortega/arteMIS.git
```

With conda:

```bash
conda env create -f environment.yml
conda activate artemis
```

Check that everything is ok:

```bash
artemis-run --help
```

Without conda:

```bash
pip install -e .
```

### Running

Check `config.yaml` and update the input/output paths before running. Default mode is `global`:

```bash
artemis-run --config config.yaml --mode global
```

---

## Input

### Global mode
- `mgf_file`: input spectra file
- `chem_file`: MS2Query results for global mode (a column such as `npc_pathway_results` set via `chem_level` in `config.yaml`)
- `out_dir`: name of your output directory — all results are saved as a folder here

### Seed mode
- `mgf_file`: input spectra file
- `chem_file`: table of the subset of annotated features (a column such as `pathway_level` set via `chem_level` — this is what the chemical optimisation focuses on)
- `out_dir`: name of your output directory — all results are saved as a folder here

### Target mode
- `mgf_file`: input spectra file
- `chem_file`: any table (MS2Query or otherwise) with the full features, with a column identifying the class to target and optimise for (see `config.yaml`)
- `out_dir`: name of your output directory — all results are saved as a folder here

---

## Outputs

- Cleaned spectra
- LHS (Latin Hypercube Sampling) of the variable space exploration
- The top 3 graphs obtained with arteMIS
- The results from all networks, in a JSON file
- `top_configs.csv` — the top 10 configurations, with their parameters and metrics

---

## Attribution

### License

The code in this package is licensed under the [MIT License](LICENSE).

### Citation

Coming soon.

### Contact

Please open a GitHub Issue for bugs/feature requests.

Maintainers: Rosina Torres-Ortega ([rosina.torresortega@wur.nl](mailto:rosina.torresortega@wur.nl)) and Esteban Charria-Girón ([esteban.charriagiron@wur.nl](mailto:esteban.charriagiron@wur.nl))
