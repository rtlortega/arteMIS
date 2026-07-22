import networkx as nx

from artemis.evaluation.topology_metrics import calculate_isolated_nodes
from artemis.evaluation.topology_metrics import calculate_degree_cv
from artemis.evaluation.topology_metrics import calculate_component_size_gini


# ---------------------------------------------------------------------------
# calculate_isolated_nodes
# ---------------------------------------------------------------------------


def test_calculate_isolated_nodes():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (4, 5)])
    G.add_node(6)  # Isolated node
    assert calculate_isolated_nodes(G) == 1 / 6  # 1 isolated out of 6 total

    G.add_node(7)  # Another isolated node
    assert calculate_isolated_nodes(G) == 2 / 7  # 2 isolated out of 7 total


def test_calculate_isolated_nodes_empty_graph():
    G = nx.Graph()
    try:
        calculate_isolated_nodes(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


# ---------------------------------------------------------------------------
# calculate_degree_cv
# ---------------------------------------------------------------------------


def test_calculate_degree_cv_empty_graph():
    G = nx.Graph()
    try:
        calculate_degree_cv(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_degree_cv_homogeneous_graph():
    # Every node in a cycle graph has degree 2 -> std == 0 -> cv == 0.0
    G = nx.cycle_graph(5)
    assert (
        calculate_degree_cv(G) == 0.0
    ), "A graph with uniform degree should have CV of 0"


def test_calculate_degree_cv_all_isolated_nodes():
    # Mean degree is 0 -> function should short-circuit to 0.0, not divide by zero
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    assert calculate_degree_cv(G) == 0.0, "CV should be 0.0 when mean degree is 0"


def test_calculate_degree_cv_value_range():
    # Hub-and-spoke: one hub connected to several leaves -> heterogeneous degrees
    G = nx.star_graph(5)  # node 0 has degree 5, nodes 1-5 have degree 1
    cv = calculate_degree_cv(G)
    assert (
        0 <= cv < 1
    ), "Degree CV should be bounded in [0, 1) by the cv/(1+cv) transform"
    assert cv > 0, "A hub-and-spoke graph should have a nonzero degree CV"


# ---------------------------------------------------------------------------
# calculate_component_size_gini
# ---------------------------------------------------------------------------


def test_calculate_component_size_gini_empty_graph():
    G = nx.Graph()
    try:
        calculate_component_size_gini(G)
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_component_size_gini_single_component():
    # One connected component containing all nodes -> Gini's n==1 special case -> 0.0
    G = nx.path_graph(5)
    assert (
        calculate_component_size_gini(G) == 0.0
    ), "A single component should have a Gini of 0.0"


def test_calculate_component_size_gini_equal_sized_components():
    # Three disconnected pairs -> all components the same size -> Gini == 0.0
    G = nx.Graph()
    G.add_edges_from([(1, 2), (3, 4), (5, 6)])
    assert (
        calculate_component_size_gini(G) == 0.0
    ), "Equally sized components should have a Gini of 0.0"


def test_calculate_component_size_gini_value_range():
    # One larger component plus a few isolated nodes -> unequal component sizes
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    G.add_nodes_from([6, 7, 8])  # isolated nodes -> components of size 1
    gini = calculate_component_size_gini(G)
    assert 0 <= gini <= 1, "Gini coefficient should be between 0 and 1"
    assert gini > 0, "Unequal component sizes should produce a nonzero Gini"
