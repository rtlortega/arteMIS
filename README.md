<p align="center">
  <img src="docs/artemis_logo_white.png" alt="arteMIS logo" width="220"/>
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

Everything about a specific **run** is passed on the command line — inputs,
output, mode, scores, tolerance, how many networks to sample, and the
chem-level column. Default mode is `global`:

```bash
# Global (default) — MS2Query results, SMILES already in --chem
artemis-run --mgf spectra.mgf --chem ms2query_results.csv --out results/ --n-networks 100

# Seed — optimise on the annotated subset
artemis-run --mode seed --mgf spectra.mgf --chem annotations.csv --out results/

# Target — --target-class implies --mode target
artemis-run --target-class melleolides --mgf spectra.mgf --chem class_table.csv --out results/

# SIRIUS-style: SMILES in a separate file, merged on feature id
artemis-run --mgf spectra.mgf --chem canopus_summary.tsv --smiles structures.tsv \
    --chem-level "NPC#pathway" --out results/

# Two score families with a wider tolerance
artemis-run --mgf spectra.mgf --chem ms2query_results.csv --out results/ \
    --scores modcosine,cosine --tolerance 0.05
```

Run-level flags: `--mgf --chem [--smiles] --out --mode --target-class
--chem-level --scores --tolerance --n-networks`. Score families available:
`modcosine`, `cosine`.

#### What the config file is for

The **method** — *how* the pipeline behaves — lives in a per-mode config under
`artemis/configs/` (`global.yaml`, `seed.yaml`, `target.yaml`). Each config
holds exactly three things:

1. **preprocessing** — `clean_peaks: true` applies high cleaning; `false` only
   harmonizes the spectra (no extra cleaning).
2. **sampling** — the Latin-hypercube **parameter ranges** explored, plus the
   `seed` for reproducibility. (How *many* networks are sampled is `--n-networks`
   on the CLI, not here.)
3. **optimization** — the **metrics** to maximize / minimize (these differ by
   mode, which is why each mode has its own file).

Each mode loads its own config automatically; pass `--config my_method.yaml`
to use a customised copy. The effective parameters and the exact command are
saved to `run_info.yaml` in the output directory.

---

## Input

Paths are passed on the command line (`--mgf`, `--chem`, optional `--smiles`,
`--out`). `--chem-level` selects the column driving the chemical optimisation
(default `npc_pathway_results` for MS2Query; use `NPC#pathway` for SIRIUS/CANOPUS).
`--chem` and `--smiles` accept `.csv` or `.tsv`.

### Global mode
- `--mgf`: input spectra file
- `--chem`: MS2Query results (class column selected by `--chem-level`, e.g. `npc_pathway_results`)
- `--out`: output directory — all results are saved as a folder here

### Seed mode
- `--mgf`: input spectra file
- `--chem`: table of the subset of annotated features (class column via `--chem-level`; rows equal to `unlabeled_value` in `seed.yaml` are dropped)
- `--out`: output directory

### Target mode
- `--mgf`: input spectra file
- `--chem`: any table (MS2Query or otherwise) with the full features and a column identifying the class
- `--target-class`: the class to target and optimise for (implies `--mode target`)
- `--out`: output directory

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
