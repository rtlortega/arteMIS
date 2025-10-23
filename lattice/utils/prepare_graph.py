import networkx as nx
import pandas as pd


def prepare_graph_class(
    G: nx.Graph, df: pd.DataFrame, feature_col: str, attribute: str
) -> nx.Graph:
    """
    Prepare the graph by adding chemical class attributes to nodes.

    Parameters:
    G (networkx.Graph): The graph to prepare.
    df (pd.DataFrame): DataFrame containing 'feature_id' and 'second_level_class'.

    Returns:
    networkx.Graph: The prepared graph with chemical class attributes.
    """
    if G is None or df is None:
        raise ValueError("Graph and DataFrame must not be None.")

    attributes_key = dict(zip(df[feature_col], df[attribute]))
    attributes_key_str = {str(k): v for k, v in attributes_key.items()}

    for node, chem_class in attributes_key_str.items():
        if node in G.nodes:
            G.nodes[node][attribute] = chem_class

    return G


def prepare_graph_fps(
    G: nx.Graph, df: pd.DataFrame, feature_col: str, attribute: str
) -> nx.Graph:
    """
    Prepare the graph by adding fingerprint attributes to nodes.

    Parameters:
    G (networkx.Graph): The graph to prepare.
    df (pd.DataFrame): DataFrame containing 'feature_id' and 'fingerprint'.

    Returns:
    networkx.Graph: The prepared graph with fingerprint attributes.
    """
    if G is None or df is None:
        raise ValueError("Graph and DataFrame must not be None.")

    attributes_key = dict(zip(df[feature_col], df[attribute]))
    attributes_key_str = {str(k): v for k, v in attributes_key.items()}

    for node, fps in attributes_key_str.items():
        if node in G.nodes:
            G.nodes[node][attribute] = fps

    return G
