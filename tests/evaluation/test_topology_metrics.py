import networkx as nx

from lattice.evaluation.topology_metrics import calculate_average_degree
from lattice.evaluation.topology_metrics import calculate_connected_nodes
from lattice.evaluation.topology_metrics import calculate_isolated_nodes
from lattice.evaluation.topology_metrics import network_component_size_metric


def test_calculate_average_degree():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
    assert calculate_average_degree(G) == 2.0

    G.add_node(5)  # Isolated node
    assert calculate_average_degree(G) == 1.6


def test_calculate_average_degree_empty_graph():
    G = nx.Graph()
    try:
        calculate_average_degree(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_connected_nodes():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    G.add_node(6)  # Isolated node
    assert calculate_connected_nodes(G) == (5 / 6) * 100  # 5 connected out of 6 total

    G.add_node(7)  # Another isolated node
    assert calculate_connected_nodes(G) == (5 / 7) * 100  # 5 connected out of 7 total


def test_calculate_connected_nodes_empty_graph():
    G = nx.Graph()
    try:
        calculate_connected_nodes(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_isolated_nodes():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    G.add_node(6)  # Isolated node
    assert calculate_isolated_nodes(G) == (1 / 6) * 100  # 1 isolated out of 6 total

    G.add_node(7)  # Another isolated node
    assert calculate_isolated_nodes(G) == (2 / 7) * 100  # 2 isolated out of 7 total


def test_calculate_isolated_nodes_empty_graph():
    G = nx.Graph()
    try:
        calculate_isolated_nodes(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_network_component_size_metric():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])  # Component of size 3
    G.add_edges_from([(4, 5)])  # Component of size 2
    G.add_node(6)  # Isolated node, component of size 1

    assert network_component_size_metric(G, 0.5) == 3  # 50% of 6 nodes is 3
    assert network_component_size_metric(G, 1 / 3) == 3  # ~33% of 6 nodes is 2
    assert network_component_size_metric(G, 1 / 6) == 3  # ~16.67% of 6 nodes is 1
