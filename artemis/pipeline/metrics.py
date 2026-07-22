"""Registry of metrics produced by the arteMIS optimisation pipeline.

Topology metrics are the same for every mode. Chemistry metrics depend on the
mode: global/seed compare intra/inter-component similarity against a class
column, target compares network structure against one specific class.

`available_metric_columns(mode)` is the single source of truth used to
validate `optimization.maximize` / `optimization.minimize` in config.yaml
before the (expensive) network-building loop runs.
"""
from artemis.evaluation.chemistry_metrics import (
    calculate_component_purity_target_components,
    calculate_consistency_measurement,
    calculate_edge_purity_target_incident,
    calculate_intra_inter_similarity,
    calculate_target_component_purity,
)
from artemis.evaluation.topology_metrics import (
    calculate_component_size_gini,
    calculate_degree_cv,
    calculate_isolated_nodes,
)

TOPOLOGY_PREFIX = "top_"
CHEMISTRY_PREFIX = "chem_"

TOPOLOGY_METRIC_KEYS = ("degree_cv", "isolated_nodes", "component_size_gini")

CHEMISTRY_METRIC_KEYS = {
    "global": ("net_avg_intra", "net_avg_inter", "consistency_measurement"),
    "seed": ("net_avg_intra", "net_avg_inter", "consistency_measurement"),
    "target": (
        "edge_purity_target_incident",
        "component_purity_target_components",
        "target_component_purity",
    ),
}


def available_metric_columns(mode: str) -> set:
    """Column names usable in optimization.maximize / .minimize for `mode`."""
    cols = {f"{TOPOLOGY_PREFIX}{k}" for k in TOPOLOGY_METRIC_KEYS}
    cols |= {f"{CHEMISTRY_PREFIX}{k}" for k in CHEMISTRY_METRIC_KEYS[mode]}
    return cols


def compute_topology_metrics(G) -> dict:
    return {
        "degree_cv": round(calculate_degree_cv(G), 4),
        "isolated_nodes": round(calculate_isolated_nodes(G), 4),
        "component_size_gini": calculate_component_size_gini(G),
    }


def compute_chemistry_metrics(
    mode: str,
    df,
    G,
    *,
    key: str = "component",
    attribute: str,
    target_class: str = None,
) -> dict:
    if mode in ("global", "seed"):
        net_avg_intra, net_avg_inter = calculate_intra_inter_similarity(df, key)
        return {
            "net_avg_intra": net_avg_intra,
            "net_avg_inter": net_avg_inter,
            "consistency_measurement": calculate_consistency_measurement(
                G, key=key, attribute=attribute
            ),
        }
    if mode == "target":
        return {
            "edge_purity_target_incident": calculate_edge_purity_target_incident(
                G,
                attribute=attribute,
                target_class=target_class,
                require_both_labeled=True,
            ),
            "component_purity_target_components": calculate_component_purity_target_components(
                G,
                component_key=key,
                class_attr=attribute,
                target_class=target_class,
                ignore_unlabeled=True,
                weight_by_target_nodes=True,
                min_component_size=2,
            ),
            "target_component_purity": calculate_target_component_purity(
                G,
                component_key=key,
                class_attr=attribute,
                target_class=target_class,
                ignore_unlabeled=True,
                min_component_size=2,
            ),
        }
    raise ValueError(f"Unknown mode: {mode!r}")
