# lattice/evaluation/run.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import networkx as nx
import pandas as pd

from lattice.networking.build import build_similarity_graph

from lattice.evaluation.chemistry_metrics import (
    calculate_edge_purity,
    calculate_component_purity,
    calculate_network_accuracy_score,
    calculate_consistency_measurement,
    calculate_edge_purity_target_incident,
    calculate_component_purity_target_components,
    calculate_target_component_purity,
)
from lattice.evaluation.topology_metrics import (
    calculate_average_degree,
    calculate_isolated_nodes,
    calculate_n20,
)


# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class EvalSpec:
    """
    Specification for evaluating a config.

    component_key:
        node attribute containing component assignment (e.g., "component")
    class_attr:
        node attribute containing chemical class labels (e.g., "library_npclassifier_pathway")
    target_class:
        if set, compute target-conditioned metrics
    """
    component_key: str = "component"
    class_attr: str = "library_npclassifier_pathway"
    target_class: Optional[str] = None


# ----------------------------
# Core evaluation
# ----------------------------

def _safe_float(x: Any) -> float:
    """Convert to float, falling back to NaN if conversion fails."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def evaluate_one_config(
    df_nodes: pd.DataFrame,
    config: Dict[str, Any],
    spec: EvalSpec,
    *,
    id_col: str = "id",
    fingerprint_col: str = "fingerprint",
    in_place_graph_node_attrs: bool = False,
) -> Dict[str, Any]:
    """
    Build a similarity graph for a single configuration and compute metrics.

    Parameters
    ----------
    df_nodes:
        Node table containing at minimum:
          - id_col (unique node IDs)
          - spec.class_attr (class labels)
          - fingerprint_col (RDKit fingerprints) if you want accuracy score
        Additional columns used by build_similarity_graph may be required.
    config:
        Parameters passed to build_similarity_graph (thresholds, score, kNN, etc.)
    spec:
        What to evaluate (global vs target-class)
    id_col:
        Column name for node IDs in df_nodes
    fingerprint_col:
        Column name for RDKit fingerprints (optional, but used by accuracy metric)
    in_place_graph_node_attrs:
        If True, allows build_similarity_graph to reuse attrs without copying (tiny speedup).
        Keep False for safety.

    Returns
    -------
    dict of metrics + config fields
    """
    if df_nodes.empty:
        raise ValueError("df_nodes is empty.")

    # 1) Build graph
    # IMPORTANT: this assumes build_similarity_graph knows how to use df_nodes + config
    G: nx.Graph = build_similarity_graph(
        df_nodes=df_nodes,
        config=config,
        id_col=id_col,
        in_place_node_attrs=in_place_graph_node_attrs,
    )

    # 2) Ensure component labels exist (if build step doesn't assign them, you must add it here)
    # If your pipeline assigns components elsewhere, remove or adapt.
    if spec.component_key not in next(iter(G.nodes(data=True)), (None, {}))[1]:
        # fallback: connected components as component ids
        comp_map = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for n in comp:
                comp_map[n] = i
        nx.set_node_attributes(G, comp_map, spec.component_key)

    # 3) Metrics (topology)
    metrics: Dict[str, Any] = {}
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["avg_degree"] = _safe_float(calculate_average_degree(G))
    metrics["isolated_nodes_frac"] = _safe_float(calculate_isolated_nodes(G))  # expects fraction in your impl
    metrics["n20"] = _safe_float(calculate_n20(G))

    # 4) Metrics (chemistry/global)
    # These require class labels to be present; the functions should gracefully ignore unlabeled
    metrics["edge_purity"] = _safe_float(calculate_edge_purity(G, spec.class_attr))
    metrics["component_purity"] = _safe_float(calculate_component_purity(G, spec.component_key, spec.class_attr))
    metrics["consistency_07"] = _safe_float(calculate_consistency_measurement(G, spec.component_key, spec.class_attr))

    # Fingerprint-based accuracy (optional but useful)
    # If fingerprints are missing, the function currently warns and skips edges.
    metrics["network_accuracy_score"] = _safe_float(calculate_network_accuracy_score(G))

    # 5) Metrics (target-conditioned)
    if spec.target_class is not None:
        metrics["edge_purity_target_incident"] = _safe_float(
            calculate_edge_purity_target_incident(G, spec.class_attr, spec.target_class)
        )
        metrics["component_purity_target_components"] = _safe_float(
            calculate_component_purity_target_components(G, spec.component_key, spec.class_attr, spec.target_class)
        )
        metrics["target_component_purity"] = _safe_float(
            calculate_target_component_purity(G, spec.component_key, spec.class_attr, spec.target_class)
        )

    # 6) Attach config fields so you can rank & reproduce
    out = {}
    out.update(config)
    out.update(metrics)
    out["target_class"] = spec.target_class
    out["class_attr"] = spec.class_attr
    out["component_key"] = spec.component_key
    return out


def evaluate_configs(
    df_nodes: pd.DataFrame,
    configs: Iterable[Dict[str, Any]],
    spec: EvalSpec,
    *,
    id_col: str = "id",
    fingerprint_col: str = "fingerprint",
    fail_fast: bool = False,
) -> pd.DataFrame:
    """
    Evaluate multiple configs and return a tidy DataFrame.

    fail_fast:
        if True, any failing config raises immediately.
        if False, failures are recorded with an 'error' column.
    """
    rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs):
        try:
            row = evaluate_one_config(
                df_nodes=df_nodes,
                config=cfg,
                spec=spec,
                id_col=id_col,
                fingerprint_col=fingerprint_col,
            )
            row["error"] = None
        except Exception as e:
            if fail_fast:
                raise
            row = dict(cfg)
            row.update({
                "error": f"{type(e).__name__}: {e}",
                "n_nodes": float("nan"),
                "n_edges": float("nan"),
            })
            row["target_class"] = spec.target_class
            row["class_attr"] = spec.class_attr
            row["component_key"] = spec.component_key

        row["config_idx"] = idx
        rows.append(row)

    return pd.DataFrame(rows)