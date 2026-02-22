import networkx as nx
from rdkit.DataStructs import TanimotoSimilarity
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from collections import Counter
import statistics


def calculate_intra_inter_similarity(df: pd.DataFrame, key: str):
    """Calculate average intra- and inter-group Tanimoto similarities based on a specified key.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'fingerprint' and grouping key columns.
    key (str): Column name used to define groups for intra- and inter-similarity calculations
    for example 'component'.
    """
    # checks if empty df
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # keep only rows with valid fingerprints
    df_valid = df[df["fingerprint"].notna()].copy()

    intra_sims = []
    inter_sims = []

    for (i, row1), (j, row2) in combinations(df_valid.iterrows(), 2):
        fp1, fp2 = row1["fingerprint"], row2["fingerprint"]
        if fp1 is None or fp2 is None:
            continue  # skip invalid pairs

        sim = TanimotoSimilarity(fp1, fp2)
        if row1[key] == row2[key]:
            intra_sims.append(sim)
        else:
            inter_sims.append(sim)

    avg_intra = np.mean(intra_sims) if intra_sims else np.nan
    avg_inter = np.mean(inter_sims) if inter_sims else np.nan
    return avg_intra, avg_inter


def calculate_edge_purity(graph: nx.Graph, attribute: str) -> float:
    """
    Calculate the edge purity based on chemical class attributes of nodes.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    attribute (str): The node attribute to use for purity calculation
    for example 'library_npclassifier_pathway'.

    Returns:
    float: from 0 till 1. The proportion of edges that connect nodes of the same chemical class.
    """
    # checks if empty dff
    if not graph:
        raise ValueError("Input Graph is empty.")

    chemical_class = nx.get_node_attributes(graph, attribute)

    chemical_class_shared_count = 0
    chemical_class_not_shared_count = 0

    for u, v, a in graph.edges(data=True):
        # skip edges if one of the nodes has no chemical_class
        if u not in chemical_class or v not in chemical_class:
            continue

        if chemical_class[u] == chemical_class[v]:
            chemical_class_shared_count += 1
        else:
            chemical_class_not_shared_count += 1

    total = chemical_class_shared_count + chemical_class_not_shared_count
    return chemical_class_shared_count / total if total > 0 else 0.0


def calculate_component_purity(G, key: str, attribute: str) -> float:
    """
    Calculate the component purity based on chemical class attributes of nodes.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    key (str): Node attribute representing the component/group.
    attribute (str): Node attribute representing the chemical class.

    Returns:
    list of purity values for each component.
    """
    # checks if empty
    if not G:
        raise ValueError("Input Graph is empty.")

    component_groups = {}

    for node, attr in G.nodes(data=True):
        comp_value = attr.get(key)
        chemical_class = attr.get(attribute)

        if comp_value not in component_groups:
            component_groups[comp_value] = {"nodes": [], "chemical_classes": []}

        component_groups[comp_value]["nodes"].append(node)
        component_groups[comp_value]["chemical_classes"].append(chemical_class)

    purities = []

    for comp_value, group in component_groups.items():
        count_classes = Counter(group["chemical_classes"])
        most_common_classes = count_classes.most_common()
        most_common_classes_list = list(most_common_classes)
        most_common_class_mf = most_common_classes_list[0][1]
        purity = most_common_class_mf / len(group["nodes"])
        purities.append(purity)
    return statistics.mean(purities)


def calculate_network_accuracy_score(G: nx.Graph) -> float:
    """Calculate the network accuracy score based on Tanimoto similarities of fingerprints within components.
    Parameters:
    G (networkx.Graph): The graph to analyze. Nodes should have 'fingerprint' attribute.
    Returns:
    float: The network accuracy score, weighted by component sizes.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Input graph is empty.")

    components = list(nx.connected_components(G))
    total_nodes = G.number_of_nodes()
    results = []

    fps_found = (
        sum("fingerprint" in data for n, data in G.nodes(data=True)),
        "of",
        G.number_of_nodes(),
    )
    print("Fingerprints found in nodes:", fps_found)

    for component in components:
        subgraph = G.subgraph(component)
        size_comp = len(component)
        edge_count_comp = subgraph.number_of_edges()

        # Isolated node (or component with no edges)
        if edge_count_comp == 0:
            score_comp = 1

        else:
            similarity_scores = []
            for u, v in subgraph.edges():
                fps_u = subgraph.nodes[u].get("fingerprint")
                fps_v = subgraph.nodes[v].get("fingerprint")

                if fps_u is None or fps_v is None:
                    warnings.warn(
                        f"Missing fingerprint for edge ({u}, {v}); skipping this edge."
                    )
                    continue

                sim_score = TanimotoSimilarity(fps_u, fps_v)
                similarity_scores.append(sim_score)

            score_comp = (
                sum(similarity_scores) / len(similarity_scores)
                if similarity_scores
                else 1
            )

        results.append((score_comp, size_comp))

    # Weighted average across components
    total_score = (
        sum(score * size for score, size in results if not pd.isna(score)) / total_nodes
    )
    return total_score


def calculate_consistency_measurement(G: nx.Graph, key: str, attribute: str) -> float:
    """
    Calculate the component purity based on chemical class attributes of nodes.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    key (str): Node attribute representing the component/group.
    attribute (str): Node attribute representing the chemical class.

    Returns:
    list of purity values for each component.
    """
    component_groups = {}

    for node, attr in G.nodes(data=True):
        comp_value = attr.get(key)
        chemical_class = attr.get(attribute)

        if comp_value not in component_groups:
            component_groups[comp_value] = {"nodes": [], attribute: []}

        component_groups[comp_value]["nodes"].append(node)
        component_groups[comp_value][attribute].append(chemical_class)

    purities = 0
    for comp_value, group in component_groups.items():
        count_classes = Counter(group[attribute])
        most_common_classes = count_classes.most_common()
        most_common_classes_list = list(most_common_classes)
        most_common_class_mf = most_common_classes_list[0][1]
        purity = most_common_class_mf / len(group["nodes"])
        if purity >= 0.7:
            purities += 1

    ratio_correct_compo = purities / len(component_groups) if component_groups else 0
    return ratio_correct_compo
    
def calculate_edge_purity_target_incident(
    G: nx.Graph,
    attribute: str,
    target_class: str,
    require_both_labeled: bool = True,
) -> float:
    """
    Edge purity conditioned on the class of interest.
    Considers only edges where at least one endpoint is in target_class.

    Returns:
        fraction of considered edges that are target-target among all edges incident to target.
        (i.e., 1 - leakage, if labels are complete)
    """
    if not G:
        raise ValueError("Input Graph is empty.")

    cls = nx.get_node_attributes(G, attribute)

    tt = 0  # target-target
    t_any = 0  # edges incident to target (target-target + target-other)

    for u, v in G.edges():
        u_c = cls.get(u)
        v_c = cls.get(v)

        if require_both_labeled and (u_c is None or v_c is None or pd.isna(u_c) or pd.isna(v_c)):
            continue

        u_is_t = (u_c == target_class)
        v_is_t = (v_c == target_class)

        if not (u_is_t or v_is_t):
            continue  # not incident to target, skip

        t_any += 1
        if u_is_t and v_is_t:
            tt += 1

    return tt / t_any if t_any else 0.0


def calculate_component_purity_target_components(
    G: nx.Graph,
    component_key: str,
    class_attr: str,
    target_class: str,
    ignore_unlabeled: bool = True,
    weight_by_target_nodes: bool = True,
    min_component_size: int = 2,
) -> float:
    """
    Component purity conditioned on a target class.
    Only components that contain at least one target node are evaluated.

    Purity is defined as (count of most common class label in component) / (component size).

    If weight_by_target_nodes=True, components contribute proportional to how many target nodes they contain
    (recommended; prevents tiny boundary components from dominating).
    """
    if not G:
        raise ValueError("Input Graph is empty.")

    # group nodes by component id
    comp_to_nodes = {}
    for n, d in G.nodes(data=True):
        comp_id = d.get(component_key)
        if comp_id is None:
            continue
        comp_to_nodes.setdefault(comp_id, []).append(n)

    purities = []
    weights = []

    for comp_id, nodes in comp_to_nodes.items():
        # --- skip tiny components ---
        if len(nodes) < min_component_size:
            continue

        labels = []
        n_target = 0
        for n in nodes:
            c = G.nodes[n].get(class_attr)
            if c is None and ignore_unlabeled:
                continue
            labels.append(c)
            if c == target_class:
                n_target += 1

        if n_target == 0:
            continue  # not a target-containing component

        if len(labels) == 0:
            continue  # all unlabeled and we're ignoring unlabeled

        counts = Counter(labels)
        purity = counts.most_common(1)[0][1] / len(labels)

        purities.append(purity)
        weights.append(n_target if weight_by_target_nodes else 1)

    if not purities:
        return 0.0

    if weight_by_target_nodes:
        return sum(p * w for p, w in zip(purities, weights)) / sum(weights)
    return statistics.mean(purities)


def calculate_target_component_purity(
    G: nx.Graph,
    component_key: str,
    class_attr: str,
    target_class: str,
    ignore_unlabeled: bool = True,
    min_component_size: int = 2,
) -> float:
    """
    For all nodes that lie in components containing >=1 target node:
    returns (# target nodes) / (# total nodes in those components).

    High = target nodes live in mostly-target components.
    """
    if not G:
        raise ValueError("Input Graph is empty.")

    # group nodes by component id
    comp_to_nodes = {}
    for n, d in G.nodes(data=True):
        comp_id = d.get(component_key)
        if comp_id is None:
            continue
        comp_to_nodes.setdefault(comp_id, []).append(n)

    total_in_target_comps = 0
    total_target = 0

    for comp_id, nodes in comp_to_nodes.items():
        if len(nodes) < min_component_size:
            continue
        
        labels = []
        for n in nodes:
            c = G.nodes[n].get(class_attr)
            if c is None and ignore_unlabeled:
                continue
            labels.append(c)

        if not labels:
            continue

        has_target = any(c == target_class for c in labels)
        if not has_target:
            continue

        total_in_target_comps += len(labels)
        total_target += sum(c == target_class for c in labels)

    return total_target / total_in_target_comps if total_in_target_comps else 0.0