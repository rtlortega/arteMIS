import networkx as nx
import numpy as np


# def calculate_average_degree(G: nx.Graph) -> float:
#     """Calculate the average degree of the graph.
#     Parameters:
#     graph (networkx.Graph): The graph to analyze.
#     Returns:
#     float: The average degree of the graph.
#     """
#     if not G or G.number_of_nodes() == 0:
#         raise ValueError("Input Graph is empty.")

#     degree = dict(G.degree())
#     return sum(degree.values()) / len(degree)


# def calculate_connected_nodes(G: nx.Graph) -> float:
#     """Calculate the number of connected nodes in the graph.
#     Parameters:
#     G (networkx.Graph): The graph to analyze.
#     Returns:
#     float: The percentage of connected nodes in the graph.
#     """
#     if not G or G.number_of_nodes() == 0:
#         raise ValueError("Input Graph is empty.")

#     connected_nodes = G.number_of_nodes() - len(list(nx.isolates(G)))
#     percent_connected_nodes = (connected_nodes / G.number_of_nodes()) * 100
#     return float(percent_connected_nodes)


def calculate_isolated_nodes(G: nx.Graph) -> float:
    """Calculate the number of isolated nodes in the graph.
    Parameters:
    G (networkx.Graph): The graph to analyze.
    Returns:
    float: The percentage of isolated nodes in the graph, between 0 and 1.
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    return len(list(nx.isolates(G))) / len(G.nodes())


# def network_component_size_metric(G: nx.Graph, threshold: float) -> float:
#     """Calculate the size of the largest components that make up a certain percentage of the graph.
#     Parameters:
#     G (networkx.Graph): The graph to analyze.
#     threshold (float): The percentage threshold (between 0 and 1).
#     Returns:
#     float: The size of the largest components that make up the given percentage of the graph.
#     """
#     if G.number_of_nodes() == 0:
#         raise ValueError("Graph is empty")

#     total_of_nodes = G.number_of_nodes()
#     top_threshold = math.ceil(threshold * total_of_nodes)
#     cluster_sizes = [
#         len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)
#     ]

#     cumulative_size = 0
#     for size in cluster_sizes:
#         cumulative_size += size
#         if cumulative_size >= top_threshold:
#             return size


def calculate_degree_cv(G: nx.Graph) -> float:
    """Calculate the coefficient of variation (std / mean) of node degrees.

    Low CV = homogeneous degree distribution (modular network).
    High CV = hub-dominated topology (hairball risk).

    Parameters:
    G (networkx.Graph): The graph to analyze.
    Returns:
    float: Coefficient of variation of degrees. 0 if mean degree is 0.
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    degrees = np.array([d for _, d in G.degree()])
    mean_deg = degrees.mean()
    if mean_deg == 0:
        return 0.0
    cv = float(degrees.std() / mean_deg)
    return cv / (1 + cv)


# def calculate_largest_component_fraction(G: nx.Graph) -> float:
#     """Calculate the fraction of nodes in the largest connected component.

#     Close to 1 = one giant hairball dominates.
#     Close to 0 = highly fragmented (many small components).
#     Minimise for modular networks.

#     Parameters:
#     G (networkx.Graph): The graph to analyze.
#     Returns:
#     float: Largest component size / total nodes, in [0, 1].
#     """
#     if not G or G.number_of_nodes() == 0:
#         raise ValueError("Input Graph is empty.")

#     largest = max(len(c) for c in nx.connected_components(G))
#     return largest / G.number_of_nodes()


def calculate_component_size_gini(G: nx.Graph) -> float:
    """Calculate the Gini coefficient of connected component sizes.

    0 = all components are the same size (ideal modular network).
    1 = one component contains all nodes.
    Minimise for modular networks.

    Parameters:
    G (networkx.Graph): The graph to analyze.
    Returns:
    float: Gini coefficient of component sizes, in [0, 1].
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    sizes = np.array(sorted(len(c) for c in nx.connected_components(G)), dtype=float)
    n = len(sizes)
    if n == 1:
        return 0.0

    # standard Gini formula
    numerator = np.sum(np.abs(sizes[:, None] - sizes[None, :]))
    denominator = 2 * n * sizes.sum()
    return float(numerator / denominator) if denominator > 0 else 0.0
