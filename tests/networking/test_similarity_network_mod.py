import networkx as nx
import numpy as np
import pytest
from matchms import Scores

from artemis.networking.SimilarityNetworkMod import SimilarityNetworkMod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_scores(score_matrix, matches_matrix=None, score_name="CosineGreedy_score"):
    """Build a real (lightweight) matchms Scores object from plain arrays.

    No spectra/mocking needed -- Scores just wraps named 2D arrays.
    """
    data = {score_name: np.array(score_matrix, dtype=float)}
    if matches_matrix is not None:
        data["matches"] = np.array(matches_matrix)
    return Scores(data)


# ---------------------------------------------------------------------------
# create_network
# ---------------------------------------------------------------------------


def test_create_network_links_above_cutoff():
    # 4 items; only (0,1) and (2,3) are similar enough to link
    scores = make_scores(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ]
    )
    net = SimilarityNetworkMod(score_cutoff=0.7, top_n=3, max_links=3)
    net.create_network(scores, identifiers=["a", "b", "c", "d"])

    assert set(net.graph.nodes) == {"a", "b", "c", "d"}
    assert set(net.graph.edges) == {("a", "b"), ("c", "d")}
    assert net.graph["a"]["b"]["weight"] == pytest.approx(0.9)


def test_create_network_no_self_links():
    scores = make_scores([[1.0, 0.2], [0.2, 1.0]])
    net = SimilarityNetworkMod(score_cutoff=0.0, top_n=2, max_links=2)
    net.create_network(scores, identifiers=["a", "b"])

    assert not net.graph.has_edge("a", "a")


def test_create_network_respects_max_links():
    # Matrix is deliberately asymmetric (real cosine matrices are symmetric,
    # but create_network never checks that) so node 0's own row is the only
    # source of edges here: no neighbor links back and re-adds one for us.
    # That isolates max_links's effect on a single row's edge count.
    scores = make_scores(
        [
            [1.0, 0.9, 0.9, 0.9],
            [0.1, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]
    )
    net = SimilarityNetworkMod(score_cutoff=0.5, top_n=3, max_links=1)
    net.create_network(scores)

    assert net.graph.degree[0] == 1


def test_create_network_min_peaks_filters_edges():
    scores = make_scores(
        [[1.0, 0.9], [0.9, 1.0]],
        matches_matrix=[[10, 2], [2, 10]],  # off-diagonal has too few matching peaks
    )
    net = SimilarityNetworkMod(score_cutoff=0.5, top_n=2, max_links=2, min_peaks=5)
    net.create_network(scores)

    assert net.graph.number_of_edges() == 0


def test_create_network_requires_symmetric_scores():
    scores = make_scores([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2]])  # 2x3, not square
    net = SimilarityNetworkMod()
    with pytest.raises(TypeError):
        net.create_network(scores)


def test_create_network_top_n_must_be_at_least_max_links():
    scores = make_scores([[1.0, 0.5], [0.5, 1.0]])
    net = SimilarityNetworkMod(top_n=1, max_links=2)
    with pytest.raises(AssertionError):
        net.create_network(scores)


def test_create_network_keep_unconnected_nodes_false_drops_isolates():
    scores = make_scores(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    net = SimilarityNetworkMod(
        score_cutoff=0.7, top_n=2, max_links=2, keep_unconnected_nodes=False
    )
    net.create_network(scores, identifiers=["a", "b", "c"])

    assert set(net.graph.nodes) == {"a", "b"}


# ---------------------------------------------------------------------------
# min_component_size / filter_components
# ---------------------------------------------------------------------------


def test_min_component_size_removes_small_components():
    scores = make_scores(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]
    )
    net = SimilarityNetworkMod(score_cutoff=0.7, top_n=3, max_links=3)
    net.create_network(scores, identifiers=["a", "b", "c", "d"])

    net.min_component_size(min_size=2)

    assert set(net.graph.nodes) == {"a", "b"}


def test_min_component_size_before_create_network_raises():
    net = SimilarityNetworkMod()
    with pytest.raises(ValueError):
        net.min_component_size(min_size=2)


def test_filter_components_prunes_large_components():
    # fully-connected star of 4 nodes around "a"; weights differ enough
    # that pruning should peel off the weakest edges first.
    scores = make_scores(
        [
            [1.0, 0.95, 0.90, 0.72],
            [0.95, 1.0, 0.3, 0.3],
            [0.90, 0.3, 1.0, 0.3],
            [0.72, 0.3, 0.3, 1.0],
        ]
    )
    net = SimilarityNetworkMod(score_cutoff=0.7, top_n=3, max_links=3)
    net.create_network(scores, identifiers=["a", "b", "c", "d"])

    net.filter_components(max_component_size=2, cosine_delta=0.02)

    for component in nx.connected_components(net.graph):
        assert len(component) <= 2


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_reports_component_ids():
    scores = make_scores(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    net = SimilarityNetworkMod(score_cutoff=0.7, top_n=3, max_links=3)
    net.create_network(scores, identifiers=["a", "b", "c", "d"])

    df = net.to_dataframe(col_name="scan")

    assert set(df["scan"]) == {"a", "b", "c", "d"}
    # a/b are one component, c/d are another
    assert (
        df.set_index("scan").loc["a", "component"]
        == df.set_index("scan").loc["b", "component"]
    )
    assert (
        df.set_index("scan").loc["a", "component"]
        != df.set_index("scan").loc["c", "component"]
    )
