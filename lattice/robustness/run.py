import json
import numpy as np
from pathlib import Path
import networkx as nx

from lattice.networking.build import build_similarity_graph

from .metrics import (
    edge_stability_original,
    node_isolation_probability,
    neighbourhood_stability_vs_original,
    giant_component_membership_probability,
    giant_component_fraction,
)


def _f(x):
    return None if (isinstance(x, float) and np.isnan(x)) else float(x)


def _edges_to_rows(d):
    return [[u, v, _f(s)] for (u, v), s in d.items()]


def _nodes_to_rows(d):
    return [[n, _f(v)] for n, v in d.items()]


def _graph_stats(g):
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "lcc_size": 0,
            "gcf": None,
        }
    comps = list(nx.connected_components(g))
    lcc = max((len(c) for c in comps), default=0)
    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "n_components": int(len(comps)),
        "lcc_size": int(lcc),
        "gcf": float(lcc / n),
    }


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_one_config(
    scores_obj,
    score_name,
    nodes_names,
    replicates_scores,
    identifier_key,
    cut_off,
    max_links,
    max_comp_size,
    link_method="single",
    min_peaks=None,
):
    original_g = build_similarity_graph(
        scores_obj,
        score_name,
        identifier_key,
        cut_off,
        max_links,
        max_comp_size,
        link_method=link_method,
        min_peaks=min_peaks,
    )

    boot_graphs = [
        build_similarity_graph(
            sc,
            score_name,
            identifier_key,
            cut_off,
            max_links,
            max_comp_size,
            link_method=link_method,
            min_peaks=min_peaks,
        )
        for sc in replicates_scores
    ]

    edge_stab = edge_stability_original(original_g, boot_graphs)
    iso_prob, iso_present = node_isolation_probability(boot_graphs, nodes_names)
    neigh = neighbourhood_stability_vs_original(original_g, boot_graphs, nodes_names)
    gc_prob, gc_present = giant_component_membership_probability(
        boot_graphs, nodes_names
    )
    gcf = giant_component_fraction(boot_graphs)

    original_stats = _graph_stats(original_g)
    boot_n_nodes = [int(g.number_of_nodes()) for g in boot_graphs]
    boot_n_edges = [int(g.number_of_edges()) for g in boot_graphs]

    return {
        "params": {
            "identifier_key": identifier_key,
            "cut_off": cut_off,
            "max_links": max_links,
            "max_comp_size": max_comp_size,
            "link_method": link_method,
            "min_peaks": min_peaks,
        },
        "n_bootstraps": len(boot_graphs),
        "original_stats": original_stats,
        "boot_stats": {
            "n_nodes": boot_n_nodes,
            "n_edges": boot_n_edges,
            "frac_empty": float(np.mean([n == 0 for n in boot_n_nodes])),
        },
        "edge_stability_original": _edges_to_rows(edge_stab),
        "node_isolation_probability": _nodes_to_rows(iso_prob),
        "node_isolation_present": iso_present,
        "neighbourhood_stability_vs_original": _nodes_to_rows(neigh),
        "giant_component_membership_probability": _nodes_to_rows(gc_prob),
        "giant_component_present": gc_present,
        "giant_component_fraction": [_f(x) for x in gcf],
    }
