"""Loads and validates config.yaml for the arteMIS pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from artemis.pipeline.metrics import available_metric_columns

VALID_MODES = ("global", "seed", "target")
DEFAULT_TOLERANCE = 0.02
DEFAULT_SCORES = {
    "modcosine": {"label": "Modified Cosine", "tolerance": DEFAULT_TOLERANCE}
}
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
    mode_override: str | None = None,
    scores_override: str | None = None,
    n_networks_override: int | None = None,
) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    mode = mode_override or raw.get("mode") or "global"
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    scores_raw = raw.get("scores") or DEFAULT_SCORES
    scores = {
        key: {
            "label": val.get("label", key),
            "tolerance": float(val.get("tolerance", DEFAULT_TOLERANCE)),
        }
        for key, val in scores_raw.items()
    }

    if scores_override:
        wanted = [s.strip() for s in scores_override.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in scores]
        if unknown:
            raise ValueError(
                f"--scores {unknown} not defined under `scores:` in {path}. Known: {list(scores)}"
            )
        scores = {k: scores[k] for k in wanted}

    paths = raw.get("paths", {})
    target = raw.get("target", {}) or {}
    preprocessing = raw.get("preprocessing", {}) or {}
    clean_peaks = bool(preprocessing.get("clean_peaks", False))

    target_class = target.get("class") or None
    if mode == "target" and not target_class:
        raise ValueError(
            "mode 'target' requires `target.class` to be set in the config"
        )

    sampling = raw.get("sampling", {}) or {}
    param_space = ParamSpace(
        bounds=sampling.get("param_space", DEFAULT_PARAM_SPACE),
        n_networks=int(
            n_networks_override
            if n_networks_override is not None
            else sampling.get("n_networks", 100)
        ),
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
        mgf_file=paths.get("mgf_file", ""),
        chem_file=paths.get("chem_file", ""),
        out_dir=paths.get("out_dir", "OUT_DIR"),
        scores=scores,
        chem_level=raw.get("chem_level", "npc_pathway_results"),
        target_class=target_class,
        unlabeled_value=target.get("unlabeled_value", DEFAULT_UNLABELED_VALUE),
        clean_peaks=clean_peaks,
        param_space=param_space,
        optimization=OptimizationConfig(
            maximize=maximize, minimize=minimize, weights=weights
        ),
    )
