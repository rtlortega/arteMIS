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

### Spectra (`--mgf`) — always the MZmine GNPS/IIMN export

arteMIS (matchms) always reads the spectra file exported from **MZmine**: the
one named `*_iimn_gnps.mgf` (the IIMN / GNPS export). This is the **only**
spectra file the pipeline reads, in **every** mode. In the demo it is
[`demo/input/demo_spectra.mgf`](demo/input/demo_spectra.mgf).

> ⚠️ **Do not** pass a SIRIUS-oriented `.mgf` (e.g. `demo/demo_300_sirius.mgf`).
> That file is produced only to run SIRIUS in parallel and is in a format
> matchms does not handle here — it is an input *to SIRIUS*, never to arteMIS.
> The spectra and the SIRIUS annotations are two separate things.

### Chemical info (`--chem`, optional `--smiles`)

`--chem` is a table (`.csv` or `.tsv`) of features with their class labels.
`--chem-level` selects the column that drives the chemical optimisation
(default `npc_pathway_results` for MS2Query; use `NPC#pathway` for SIRIUS/CANOPUS).

The optimisation also needs a **SMILES** per feature:

- **MS2Query** keeps SMILES in the same results table, so `--chem` alone is enough.
- **SIRIUS** splits them: class predictions live in the CANOPUS structure summary,
  while SMILES live in the structure-identifications table. Pass the class table
  to `--chem` and the structures to `--smiles`; arteMIS merges them on the feature
  id (`feature_id`, or `mappingFeatureId` for SIRIUS).

### Modes

| Mode | `--chem` holds… | Intention |
|---|---|---|
| **global** | all annotated features (e.g. full MS2Query results) | use everything you have |
| **seed** | a small, high-confidence subset of annotations | start from a trusted core |
| **target** | any table with a class column (also pass `--target-class`) | build the network around one class |

**global and seed share the same method** — identical metrics and optimisation.
The only difference is *intention*: **seed** starts from a few high-confidence
annotations, **global** uses them all. (In seed mode, rows whose `--chem-level`
value equals `unlabeled_value` in `seed.yaml` are dropped.)

All runs also write to `--out`, the output directory (created if missing).

---

## Outputs

- Cleaned spectra
- LHS (Latin Hypercube Sampling) of the variable space exploration
- The top 3 graphs obtained with arteMIS
- The results from all networks, in a JSON file
- `top_configs.csv` — the top 10 configurations, with their parameters and metrics

---

## Demo

A small dataset is bundled under [`demo/`](demo/) so you can try every mode.
All runs use the same spectra file, [`demo/input/demo_spectra.mgf`](demo/input/demo_spectra.mgf)
— the MZmine `*_iimn_gnps` export (see [Input](#input)). Run the commands from
the repository root.

### 1) MS2Query input

MS2Query puts class predictions **and** SMILES in one table, so only `--chem` is needed.

```bash
# global — use all annotated features
artemis --mgf demo/input/demo_spectra.mgf \
    --chem demo/input/chem_file/demo_MS2Query.csv \
    --out demo/tests/MS2Query_global

# seed — start from the high-confidence subset
artemis --mode seed --mgf demo/input/demo_spectra.mgf \
    --chem demo/input/chem_file/MS2Query_demo_spectra_seed.csv \
    --out demo/tests/MS2Query_seed

# target — build the network around one class (quote multi-word names)
artemis --target-class "Shikimates and Phenylpropanoids" \
    --mgf demo/input/demo_spectra.mgf \
    --chem demo/input/chem_file/demo_MS2Query.csv \
    --out demo/tests/MS2Query_target
```

The seed table (`MS2Query_demo_spectra_seed.csv`) is produced by
[`demo/generate_demo_seed_chem_file.ipynb`](demo/generate_demo_seed_chem_file.ipynb),
which keeps only the MS2Query rows with `ms2query_model_prediction > 0.7`
(294 → 50 features). As explained in [Input](#input), seed and global run the
*same* method — the seed run just starts from that trusted subset.

### 2) SIRIUS input

SIRIUS splits class predictions (CANOPUS summary) from SMILES (structure
identifications), so pass both files and set `--chem-level` to the CANOPUS column:

```bash
artemis --mgf demo/input/demo_spectra.mgf \
    --chem demo/input/chem_file/demo_SIRIUS_canopus_structure_summary.csv \
    --smiles demo/input/chem_file/demo_SIRIUS_structure_identifications.csv \
    --chem-level "NPC#pathway" \
    --out demo/tests/SIRIUS_global
```

`.tsv` versions of the SIRIUS tables sit in the same folder and work too (swap
the `.csv` paths for their `.tsv` counterparts).

> Reminder: `demo/demo_300_sirius.mgf` is **not** a valid `--mgf`. It exists only
> for running SIRIUS and is never read by arteMIS — always use the MZmine
> `*_iimn_gnps` spectra file.

---

## Attribution

### License

The code in this package is licensed under the [MIT License](LICENSE).

### Citation

Coming soon.

### Contact

Please open a GitHub Issue for bugs/feature requests.

Maintainers: Rosina Torres-Ortega ([rosina.torresortega@wur.nl](mailto:rosina.torresortega@wur.nl)) and Esteban Charria-Girón ([esteban.charriagiron@wur.nl](mailto:esteban.charriagiron@wur.nl))
