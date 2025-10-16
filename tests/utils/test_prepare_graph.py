import networkx as nx
import pandas as pd

from lattice.utils.prepare_graph import prepare_graph_class
from lattice.utils.prepare_graph import prepare_graph_fps


def test_prepare_graph_class():
    # Create a simple graph
    G = nx.Graph()
    G.add_nodes_from(["1", "2", "3", "4"])
    G.add_edges_from([("1", "2"), ("2", "3")])

    # Create a DataFrame with feature_id and second_level_class
    data = {
        "feature_id": ["1", "2", "3", "5"],  # Note: '5' is not in the graph
        "second_level_class": ["A", "B", "A", "C"],
    }
    df = pd.DataFrame(data)

    # Prepare the graph
    G_prepared = prepare_graph_class(
        G, df, key="feature_id", attribute="second_level_class"
    )

    # Check if the attributes are correctly assigned
    assert G_prepared.nodes["1"]["second_level_class"] == "A"
    assert G_prepared.nodes["2"]["second_level_class"] == "B"
    assert G_prepared.nodes["3"]["second_level_class"] == "A"
    assert (
        "second_level_class" not in G_prepared.nodes["4"]
    )  # Node '4' should not have the attribute


def test_prepare_graph_fps():
    # Create a simple graph
    G = nx.Graph()
    G.add_nodes_from(["1", "2", "3", "4"])
    G.add_edges_from([("1", "2"), ("2", "3")])

    # Create a DataFrame with feature_id and fps
    data = {
        "feature_id": ["1", "2", "3", "5"],  # Note: '5' is not in the graph
        "fps": ["1111", "1110", "1111", "1100"],
    }
    df = pd.DataFrame(data)

    # Prepare the graph
    G_prepared = prepare_graph_fps(G, df, key="feature_id", attribute="fps")

    # Check if the attributes are correctly assigned
    assert G_prepared.nodes["1"]["fps"] == "1111"
    assert G_prepared.nodes["2"]["fps"] == "1110"
    assert "fps" not in G_prepared.nodes["4"]  # Node '4' should not have the attribute
