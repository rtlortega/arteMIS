import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx

from artemis.evaluation.chemistry_metrics import calculate_intra_inter_similarity
from artemis.evaluation.chemistry_metrics import calculate_consistency_measurement
from artemis.evaluation.chemistry_metrics import calculate_edge_purity_target_incident
from artemis.evaluation.chemistry_metrics import (
    calculate_component_purity_target_components,
)
from artemis.evaluation.chemistry_metrics import calculate_target_component_purity


ATTR = "library_npclassifier_pathway"


# ---------------------------------------------------------------------------
# calculate_intra_inter_similarity
# ---------------------------------------------------------------------------


def test_calculate_intra_inter_similarity():
    # Create a sample dataframe with fingerprints and components
    data = {
        "smiles": [
            "CCO",
            "CCN",
            "CCC",
            "CNC",
            "COC",
            "CCO",
        ],  # Ethanol, Ethylamine, Propane, Methylamine, Dimethyl ether, Ethanol
        "component": ["A", "A", "B", "B", "C", "A"],
    }
    df = pd.DataFrame(data)

    # Generate fingerprints
    df["fingerprint"] = df["smiles"].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(x), 2, nBits=1024
        )
    )

    # Calculate intra and inter similarities
    avg_intra, avg_inter = calculate_intra_inter_similarity(df, key="component")

    # Assertions to check if the values are within expected ranges
    assert avg_intra is not None, "Intra similarity should not be None"
    assert avg_inter is not None, "Inter similarity should not be None"
    assert 0 <= avg_intra <= 1, "Intra similarity should be between 0 and 1"
    assert 0 <= avg_inter <= 1, "Inter similarity should be between 0 and 1"
    assert (
        avg_intra >= avg_inter
    ), "Intra similarity should be greater than or equal to inter similarity"


def test_calculate_intra_inter_similarity_empty_df():
    df = pd.DataFrame(columns=["smiles", "component", "fingerprint"])
    try:
        calculate_intra_inter_similarity(df, key="component")
    except ValueError as e:
        assert str(e) == "Input DataFrame is empty."
    else:
        assert False, "Expected ValueError for empty dataframe"


def test_calculate_intra_inter_similarity_single_group():
    # All rows share one component -> there are no inter-group pairs at all,
    # so avg_inter must come back as NaN instead of e.g. raising or being 0.
    data = {"smiles": ["CCO", "CCN", "CCC"], "component": ["A", "A", "A"]}
    df = pd.DataFrame(data)
    df["fingerprint"] = df["smiles"].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(x), 2, nBits=1024
        )
    )

    avg_intra, avg_inter = calculate_intra_inter_similarity(df, key="component")

    assert not pd.isna(
        avg_intra
    ), "Intra similarity should be computed for a group with >1 member"
    assert pd.isna(
        avg_inter
    ), "Inter similarity should be NaN when there is only one group"


# ---------------------------------------------------------------------------
# calculate_consistency_measurement
# ---------------------------------------------------------------------------


def test_calculate_consistency_measurement():
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", ATTR: "A"}),
            (2, {"component": "X", ATTR: "A"}),
            (3, {"component": "X", ATTR: "B"}),
            (4, {"component": "Y", ATTR: "B"}),
            (5, {"component": "Y", ATTR: "B"}),
            (6, {"component": "Y", ATTR: "B"}),
        ]
    )
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])

    # Calculate consistency measurement
    consistency = calculate_consistency_measurement(G, key="component", attribute=ATTR)

    expected_consistency = 0.5

    # Assertions to check if the value is within expected ranges
    assert 0 <= consistency <= 1, "Consistency measurement should be between 0 and 1"
    assert (
        consistency == expected_consistency
    ), f"Expected {expected_consistency}, got {consistency}"
    assert isinstance(consistency, float), "Consistency measurement should be a float"


def test_calculate_consistency_measurement_empty_graph():
    # Unlike the other metrics below, this function has no explicit empty-graph
    # guard: total_nodes stays 0 and it should fall back to 0.0, not raise.
    G = nx.Graph()
    consistency = calculate_consistency_measurement(G, key="component", attribute=ATTR)
    assert consistency == 0.0, "Consistency measurement of an empty graph should be 0.0"


def test_calculate_consistency_measurement_purity_threshold():
    # A component with purity exactly at the 0.7 cutoff should count as
    # consistent, since the source uses `purity >= 0.7`.
    G = nx.Graph()
    G.add_nodes_from(
        [(i, {"component": "X", ATTR: "A"}) for i in range(1, 8)]
        + [(i, {"component": "X", ATTR: "B"}) for i in range(8, 11)]
    )  # 7 "A" + 3 "B" out of 10 -> purity = 0.7 exactly

    consistency = calculate_consistency_measurement(G, key="component", attribute=ATTR)
    assert (
        consistency == 1.0
    ), "A component with purity exactly at 0.7 should count as consistent"


# ---------------------------------------------------------------------------
# calculate_edge_purity_target_incident
# ---------------------------------------------------------------------------


def test_calculate_edge_purity_target_incident_empty_graph():
    G = nx.Graph()
    try:
        calculate_edge_purity_target_incident(G, attribute=ATTR, target_class="A")
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_edge_purity_target_incident_value_range():
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {ATTR: "A"}),
            (2, {ATTR: "A"}),
            (3, {ATTR: "B"}),
            (4, {ATTR: "B"}),
            (5, {}),  # unlabeled
        ]
    )
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

    purity = calculate_edge_purity_target_incident(G, attribute=ATTR, target_class="A")

    assert 0 <= purity <= 1, "Edge purity should be between 0 and 1"
    # Edges incident to "A": (1,2) A-A and (2,3) A-B -> 1 of 2 is target-target.
    # (3,4) isn't incident to the target; (4,5) is dropped since node 5 is unlabeled.
    assert purity == 0.5, f"Expected 0.5, got {purity}"


def test_calculate_edge_purity_target_incident_no_target_edges():
    # No edge is incident to the target class -> should return 0.0 instead of
    # dividing by zero (t_any == 0).
    G = nx.Graph()
    G.add_nodes_from([(1, {ATTR: "B"}), (2, {ATTR: "B"})])
    G.add_edges_from([(1, 2)])

    purity = calculate_edge_purity_target_incident(G, attribute=ATTR, target_class="A")
    assert (
        purity == 0.0
    ), "Purity should be 0.0 when no edges are incident to the target class"


# ---------------------------------------------------------------------------
# calculate_component_purity_target_components
# ---------------------------------------------------------------------------


def test_calculate_component_purity_target_components_empty_graph():
    G = nx.Graph()
    try:
        calculate_component_purity_target_components(
            G, component_key="component", class_attr=ATTR, target_class="A"
        )
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_component_purity_target_components_no_target_class():
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", ATTR: "B"}),
            (2, {"component": "X", ATTR: "B"}),
            (3, {"component": "Y", ATTR: "C"}),
            (4, {"component": "Y", ATTR: "C"}),
        ]
    )

    purity = calculate_component_purity_target_components(
        G, component_key="component", class_attr=ATTR, target_class="A"
    )
    assert (
        purity == 0.0
    ), "Purity should be 0.0 when no component contains the target class"


# ---------------------------------------------------------------------------
# calculate_target_component_purity
# ---------------------------------------------------------------------------


def test_calculate_target_component_purity_empty_graph():
    G = nx.Graph()
    try:
        calculate_target_component_purity(
            G, component_key="component", class_attr=ATTR, target_class="A"
        )
    except ValueError as e:
        assert str(e) == "Input Graph is empty."
    else:
        assert False, "Expected ValueError for empty graph"


def test_calculate_target_component_purity_all_target():
    # A component made entirely of target nodes -> ratio should be 1.0.
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", ATTR: "A"}),
            (2, {"component": "X", ATTR: "A"}),
            (3, {"component": "X", ATTR: "A"}),
        ]
    )

    purity = calculate_target_component_purity(
        G, component_key="component", class_attr=ATTR, target_class="A"
    )
    assert purity == 1.0, f"Expected 1.0, got {purity}"


def test_calculate_target_component_purity_no_target():
    # No target nodes anywhere -> no component qualifies -> 0.0.
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", ATTR: "B"}),
            (2, {"component": "X", ATTR: "B"}),
        ]
    )

    purity = calculate_target_component_purity(
        G, component_key="component", class_attr=ATTR, target_class="A"
    )
    assert purity == 0.0, "Purity should be 0.0 when there are no target nodes"
