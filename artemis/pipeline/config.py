"""Loads and validates config.yaml for the arteMIS pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from artemis.pipeline.metrics import available_metric_columns

VALID_MODES = ("global", "seed", "target")
DEFAULT_TOLERANCE = 0.02
# Score families selectable with --scores. Must match SCORE_FUNCTIONS in runner.py.
KNOWN_SCORES = ("modcosine", "cosine")
DEFAULT_UNLABELED_VALUE = "No label"
DEFAULT_PARAM_SPACE = {
    "max_comp_size": [10, 100],
    "max_links": [5, 20],
    "cut_off": [0.55, 0.80],
    "matching_peaks": [2, 8],
}


@dataclass
class ParamSpace:
    bounds: dict
    n_networks: int
    seed: int

    @property
    def max_links_upper_bound(self) -> int:
        return int(self.bounds["max_links"][1])


@dataclass
class OptimizationConfig:
    maximize: list = field(default_factory=list)
    minimize: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    mode: str
    mgf_file: str
    chem_file: str
    smiles_file: str | None
    out_dir: str
    scores: dict
    chem_level: str
    target_class: str | None
    unlabeled_value: str
    clean_peaks: bool
    param_space: ParamSpace
    optimization: OptimizationConfig


def load_config(
    path: str,
    *,
    mode: str,
    mgf_file: str,
    chem_file: str,
    out_dir: str,
    chem_level: str,
    scores: str,
    tolerance: float,
    n_networks: int,
    smiles_file: str | None = None,
    target_class: str | None = None,
) -> PipelineConfig:
    """Build a PipelineConfig for one run.

    The *method* (preprocessing, LHS sampling space, optimization metrics) is
    read from the yaml at `path`. Everything about this specific *run* — mode,
    paths, chem_level, scores, tolerance, n_networks, target_class — is passed
    in from the command line.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Paths, scores and n_networks used to live in the config; they are now
    # command-line arguments. Warn so a stale config isn't silently ignored.
    for stale, flag in (
        ("paths", "--mgf/--chem/--out"),
        ("scores", "--scores/--tolerance"),
    ):
        if stale in raw:
            print(
                f"Warning: the `{stale}:` block in {path} is ignored — it is now "
                f"passed on the command line ({flag})."
            )

    # Selected score families, all sharing the command-line tolerance.
    wanted = [s.strip() for s in scores.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in KNOWN_SCORES]
    if unknown:
        raise ValueError(
            f"--scores {unknown} not recognized. Known score families: {list(KNOWN_SCORES)}"
        )
    if not wanted:
        raise ValueError("--scores must name at least one score family")
    scores_cfg = {
        name: {"label": name, "tolerance": float(tolerance)} for name in wanted
    }

    # Normalize the optional separate SMILES source (SIRIUS). When it equals
    # chem_file or is empty, SMILES come from chem_file (e.g. MS2Query).
    smiles_file = smiles_file or None

    preprocessing = raw.get("preprocessing", {}) or {}
    clean_peaks = bool(preprocessing.get("clean_peaks", False))

    if mode == "target" and not target_class:
        raise ValueError(
            "mode 'target' requires a target class (pass --target-class NAME)"
        )

    sampling = raw.get("sampling", {}) or {}
    param_space = ParamSpace(
        bounds=sampling.get("param_space", DEFAULT_PARAM_SPACE),
        n_networks=int(n_networks),
        seed=int(sampling.get("seed", 27)),
    )

    opt_raw = raw.get("optimization", {}) or {}
    maximize = list(opt_raw.get("maximize", []) or [])
    minimize = list(opt_raw.get("minimize", []) or [])
    weights = dict(opt_raw.get("weights", {}) or {})

    valid_cols = available_metric_columns(mode)
    for group_name, group in (("maximize", maximize), ("minimize", minimize)):
        unknown = [m for m in group if m not in valid_cols]
        if unknown:
            raise ValueError(
                f"optimization.{group_name} contains metrics not available for mode={mode!r}: "
                f"{unknown}. Available: {sorted(valid_cols)}"
            )
    overlap = set(maximize) & set(minimize)
    if overlap:
        raise ValueError(
            f"Metrics cannot be in both maximize and minimize: {sorted(overlap)}"
        )

    return PipelineConfig(
        mode=mode,
        mgf_file=mgf_file,
        chem_file=chem_file,
        smiles_file=smiles_file,
        out_dir=out_dir,
        scores=scores_cfg,
        chem_level=chem_level,
        target_class=target_class,
        unlabeled_value=raw.get("unlabeled_value", DEFAULT_UNLABELED_VALUE),
        clean_peaks=clean_peaks,
        param_space=param_space,
        optimization=OptimizationConfig(
            maximize=maximize, minimize=minimize, weights=weights
        ),
    )
