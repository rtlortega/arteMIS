# robustness_metrics.py
# Minimal robustness metrics for bootstrapped (NetworkX-like) graphs.

# robustness_metrics.py
import numpy as np
import networkx as nx


def edge_stability_original(original_network, bootstrapped_networks):
    """
    Stability for ORIGINAL edges only.

    For each original edge (u, v):
      stability = (#replicas where u and v present AND edge present) /
                  (#replicas where u and v present)

    Returns: dict {(u, v): stability} with edges stored as sorted tuples.
    """
    orig_edges = set(tuple(sorted(e)) for e in original_network.edges)

    # Precompute node/edge sets for replicas (simple + reusable)
    rep_nodes = [set(g.nodes) for g in bootstrapped_networks]
    rep_edges = [set(tuple(sorted(e)) for e in g.edges) for g in bootstrapped_networks]

    out = {}
    for u, v in orig_edges:
        num = 0
        den = 0
        for ns, es in zip(rep_nodes, rep_edges):
            if (u in ns) and (v in ns):
                den += 1
                if (u, v) in es:
                    num += 1
        out[(u, v)] = (num / den) if den > 0 else np.nan

    return out


def node_isolation_probability(bootstrapped_networks, nodes_names):
    """
    For each node n:
      P(isolated | present) where isolated means degree(n)==0 in that replica.

    Returns:
      isolation_prob: {node: prob} (np.nan if never present)
      present_count : {node: #replicas present}
    """
    nodes_set = set(nodes_names)
    present = {n: 0 for n in nodes_names}
    isolated = {n: 0 for n in nodes_names}

    for g in bootstrapped_networks:
        # only nodes that are present AND tracked
        present_nodes = set(g.nodes) & nodes_set
        for n in present_nodes:
            present[n] += 1
            if g.degree(n) == 0:
                isolated[n] += 1

    prob = {
        n: (isolated[n] / present[n]) if present[n] > 0 else np.nan for n in nodes_names
    }
    return prob, present


def neighbourhood_stability_vs_original(
    original_network, bootstrapped_networks, nodes_names
):
    """
    Per node, mean Jaccard similarity of neighbors vs original.

    For each replica where node is present:
      J = |N_orig ∩ N_rep| / |N_orig ∪ N_rep|
    If both neighbor sets empty -> 1.0

    Returns: {node: mean_jaccard} (np.nan if node never appears)
    """
    nodes_set = set(nodes_names)
    orig_neighbors = {
        n: set(original_network.neighbors(n)) if n in original_network else set()
        for n in nodes_names
    }
    vals = {n: [] for n in nodes_names}

    for g in bootstrapped_networks:
        present_nodes = set(g.nodes) & nodes_set
        for n in present_nodes:
            rep_nei = set(g.neighbors(n))
            orig_nei = orig_neighbors[n]
            union = orig_nei | rep_nei
            vals[n].append(
                1.0 if len(union) == 0 else len(orig_nei & rep_nei) / len(union)
            )

    return {n: (float(np.mean(vals[n])) if vals[n] else np.nan) for n in nodes_names}


def giant_component_membership_probability(bootstrapped_networks, nodes_names):
    """
    For each node n:
      P(n in giant component | n present)

    Giant component is computed as the largest connected component (treating edges undirected).

    Returns:
      prob_in_gc  : {node: prob} (np.nan if never present)
      present_cnt : {node: #replicas present}
    """
    nodes_set = set(nodes_names)
    present = {n: 0 for n in nodes_names}
    in_gc = {n: 0 for n in nodes_names}

    for g in bootstrapped_networks:
        present_nodes = set(g.nodes) & nodes_set
        for n in present_nodes:
            present[n] += 1

        if g.number_of_nodes() == 0:
            continue
        lcc = max(nx.connected_components(g), key=len, default=set())
        for n in lcc & nodes_set:
            in_gc[n] += 1

    prob = {
        n: (in_gc[n] / present[n]) if present[n] > 0 else np.nan for n in nodes_names
    }
    return prob, present


def giant_component_fraction(bootstrapped_networks):
    """
    For each replicate graph g:
      GCF = |largest connected component| / |V|
    Returns:
      gcf_list : list[float] length = #replicates (np.nan if graph empty)
    """
    gcf = []
    for g in bootstrapped_networks:
        n = g.number_of_nodes()
        if n == 0:
            gcf.append(np.nan)
            continue
        lcc = max((len(c) for c in nx.connected_components(g)), default=0)
        gcf.append(lcc / n)
    return gcf
