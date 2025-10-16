import networkx as nx
import math


def calculate_average_degree(G: nx.Graph) -> float:
    """Calculate the average degree of the graph.
    Parameters:
    graph (networkx.Graph): The graph to analyze.
    Returns:
    float: The average degree of the graph.
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    degree = dict(G.degree())
    return sum(degree.values()) / len(degree)


def calculate_connected_nodes(G: nx.Graph) -> float:
    """Calculate the number of connected nodes in the graph.
    Parameters:
    G (networkx.Graph): The graph to analyze.
    Returns:
    float: The percentage of connected nodes in the graph.
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    connected_nodes = G.number_of_nodes() - len(list(nx.isolates(G)))
    percent_connected_nodes = (connected_nodes / G.number_of_nodes()) * 100
    return float(percent_connected_nodes)


def calculate_isolated_nodes(G: nx.Graph) -> float:
    """Calculate the number of isolated nodes in the graph.
    Parameters:
    G (networkx.Graph): The graph to analyze.
    Returns:
    float: The percentage of isolated nodes in the graph.
    """
    if not G or G.number_of_nodes() == 0:
        raise ValueError("Input Graph is empty.")

    return (len(list(nx.isolates(G))) / len(G.nodes())) * 100


def network_component_size_metric(G: nx.Graph, threshold: float) -> float:
    """Calculate the size of the largest components that make up a certain percentage of the graph.
    Parameters:
    G (networkx.Graph): The graph to analyze.
    threshold (float): The percentage threshold (between 0 and 1).
    Returns:
    float: The size of the largest components that make up the given percentage of the graph.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty")

    total_of_nodes = G.number_of_nodes()
    top_threshold = math.ceil(threshold * total_of_nodes)
    cluster_sizes = [
        len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]

    cumulative_size = 0
    for size in cluster_sizes:
        cumulative_size += size
        if cumulative_size >= top_threshold:
            return size
