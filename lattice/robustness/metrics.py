# robustness_metrics.py
# Minimal robustness metrics for bootstrapped (NetworkX-like) graphs.

# robustness_metrics.py
import numpy as np


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
    present = {n: 0 for n in nodes_names}
    isolated = {n: 0 for n in nodes_names}

    for g in bootstrapped_networks:
        node_set = set(g.nodes)
        for n in nodes_names:
            if n in node_set:
                present[n] += 1
                if g.degree(n) == 0:
                    isolated[n] += 1

    prob = {}
    for n in nodes_names:
        prob[n] = (isolated[n] / present[n]) if present[n] > 0 else np.nan

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
    orig_neighbors = {
        n: (set(original_network.neighbors(n)) if n in original_network else set())
        for n in nodes_names
    }

    vals = {n: [] for n in nodes_names}

    for g in bootstrapped_networks:
        node_set = set(g.nodes)
        for n in nodes_names:
            if n not in node_set:
                continue

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
    present = {n: 0 for n in nodes_names}
    in_gc = {n: 0 for n in nodes_names}

    for g in bootstrapped_networks:
        node_set = set(g.nodes)
        for n in nodes_names:
            if n in node_set:
                present[n] += 1

        # compute GC nodes (inline, no extra helper function)
        unvisited = set(node_set)
        best = set()
        while unvisited:
            start = next(iter(unvisited))
            stack = [start]
            comp = {start}
            unvisited.remove(start)

            while stack:
                u = stack.pop()
                for v in g.neighbors(u):
                    if v in unvisited:
                        unvisited.remove(v)
                        comp.add(v)
                        stack.append(v)

            if len(comp) > len(best):
                best = comp

        for n in best:
            if n in in_gc:  # only count nodes we track
                in_gc[n] += 1

    prob = {}
    for n in nodes_names:
        prob[n] = (in_gc[n] / present[n]) if present[n] > 0 else np.nan

    return prob, present
