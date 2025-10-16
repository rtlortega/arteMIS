import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import warnings

from scr.evaluation.chemistry_metrics import calculate_intra_inter_similarity
from scr.evaluation.chemistry_metrics import calculate_edge_purity
from scr.evaluation.chemistry_metrics import calculate_component_purity
from scr.evaluation.chemistry_metrics import calculate_network_accuracy_score
from scr.evaluation.chemistry_metrics import calculate_consistency_measurement


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

    print(f"Average Intra Similarity: {avg_intra}")
    print(f"Average Inter Similarity: {avg_inter}")

    # Assertions to check if the values are within expected ranges
    assert avg_intra is not None, "Intra similarity should not be None"
    assert avg_inter is not None, "Inter similarity should not be None"
    assert 0 <= avg_intra <= 1, "Intra similarity should be between 0 and 1"
    assert 0 <= avg_inter <= 1, "Inter similarity should be between 0 and 1"
    assert (
        avg_intra >= avg_inter
    ), "Intra similarity should be greater than or equal to inter similarity"


def test_calculate_edge_purity():
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"library_npclassifier_pathway": "A"}),
            (2, {"library_npclassifier_pathway": "A"}),
            (3, {"library_npclassifier_pathway": "B"}),
            (4, {"library_npclassifier_pathway": "B"}),
        ]
    )
    G.add_edges_from([(1, 2), (3, 4), (1, 3)])

    # Calculate edge purity
    purity = calculate_edge_purity(G, attribute="library_npclassifier_pathway")

    print(f"Edge Purity: {purity}")

    # Assertions to check if the value is within expected ranges
    assert 0 <= purity <= 1, "Edge purity should be between 0 and 1"
    assert purity == 2 / 3, "Edge purity should be 2/3 for this graph"
    assert isinstance(purity, float), "Edge purity should be a float"


def test_calculate_component_purity():
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", "library_npclassifier_pathway": "A"}),
            (2, {"component": "X", "library_npclassifier_pathway": "A"}),
            (3, {"component": "X", "library_npclassifier_pathway": "B"}),
            (4, {"component": "Y", "library_npclassifier_pathway": "B"}),
            (5, {"component": "Y", "library_npclassifier_pathway": "B"}),
            (6, {"component": "Y", "library_npclassifier_pathway": "C"}),
        ]
    )
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])

    # Calculate component purity
    purity = calculate_component_purity(
        G, key="component", attribute="library_npclassifier_pathway"
    )

    print(f"Component Purity: {purity}")

    # Assertions to check if the value is within expected ranges
    assert 0 <= purity <= 1, "Component purity should be between 0 and 1"
    assert isinstance(purity, float), "Component purity should be a float"
    assert (
        abs(purity - 2 / 3) < 1e-6
    ), "Component purity should be approximately 2/3 for this graph"


def test_calculate_network_accuracy_score():
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(
        [
            (
                1,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("CCO"), 2, nBits=1024
                    )
                },
            ),  # Ethanol
            (
                2,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("CCN"), 2, nBits=1024
                    )
                },
            ),  # Ethylamine
            (
                3,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("CCC"), 2, nBits=1024
                    )
                },
            ),  # Propane
            (4, {"fingerprint": None}),  # Missing fingerprint
            (
                5,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("COC"), 2, nBits=1024
                    )
                },
            ),  # Dimethyl ether
            (
                6,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("CCO"), 2, nBits=1024
                    )
                },
            ),  # Ethanol
            (
                7,
                {
                    "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles("CCO"), 2, nBits=1024
                    )
                },
            ),  # Isolated node
        ]
    )
    G.add_edges_from([(1, 2), (1, 6), (3, 4), (5, 6)])

    # Capture warnings for missing fingerprints
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score = calculate_network_accuracy_score(G)

        # Check that a warning was raised for missing fingerprints
        assert any(
            "Missing fingerprint" in str(warn.message) for warn in w
        ), "Expected warning for missing fingerprints"

    # Basic assertions
    assert isinstance(score, float), "Score should be a float"
    assert 0 <= score <= 1, "Score should be between 0 and 1"

    # Check isolated node contributes correctly
    # Node 7 is isolated, so it should not break calculation and contributes size weight
    assert score > 0, "Score should be positive"


def test_calculate_consistency_measurement():
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(
        [
            (1, {"component": "X", "library_npclassifier_pathway": "A"}),
            (2, {"component": "X", "library_npclassifier_pathway": "A"}),
            (3, {"component": "X", "library_npclassifier_pathway": "B"}),
            (4, {"component": "Y", "library_npclassifier_pathway": "B"}),
            (5, {"component": "Y", "library_npclassifier_pathway": "B"}),
            (6, {"component": "Y", "library_npclassifier_pathway": "B"}),
        ]
    )
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])

    # Calculate consistency measurement
    consistency = calculate_consistency_measurement(
        G, key="component", attribute="library_npclassifier_pathway"
    )

    expected_consistency = 0.5

    # Assertions to check if the value is within expected ranges
    assert 0 <= consistency <= 1, "Consistency measurement should be between 0 and 1"
    assert (
        consistency == expected_consistency
    ), f"Expected {expected_consistency}, got {consistency}"
    assert isinstance(consistency, float), "Consistency measurement should be a float"
