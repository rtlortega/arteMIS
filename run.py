# General Importing
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

from matchms import calculate_scores
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.importing import load_from_mgf
from matchms.networking.networking_functions import get_top_hits
from matchms.similarity.FlashSimilarity import FlashSimilarity

from artemis.evaluation.chemistry_metrics import (
    calculate_component_purity,
    calculate_consistency_measurement,
    calculate_edge_purity,
    calculate_edge_purity_target_incident,
    calculate_component_purity_target_components,
    calculate_intra_inter_similarity,
    calculate_network_accuracy_score,
    calculate_target_component_purity,
)
from artemis.evaluation.topology_metrics import (
    calculate_average_degree,
    calculate_isolated_nodes,
    network_component_size_metric,
)
from artemis.networking.SimilarityNetworkMod import SimilarityNetworkMod
from artemis.utils.fps import smiles_to_morgan_fps
from artemis.utils.lhs import get_latin_hypercube_samples
from artemis.utils.prepare_graph import prepare_graph_class, prepare_graph_fps

t0 = time.time()

# -------------------------
# To change each run:
# -------------------------
MGF_FILE = "/Users/rtlortega/Documents/PhD/Secondment/matteosimone/Downloads/ARTEMIS_project/ARTEMIS_101.mgf"
CHEM_FILE = (
    "/Users/rtlortega/Documents/PhD/Secondment/matteosimone/MS2Query_ARTEMIS_101.csv"
)
CLEANED_SPECTRA_FILE = "test_cleaned_spectra.mgf"
SCORES_FILE = "test_scores.json"
LHS_FILE = "test_LHS.json"
NETWORKS_OUTPUT_FILE = "test_networks.json"
TOP_CONFIGS_FILE = "test_top_configs.csv"
TARGET_CHEM_LEVEL = "npc_pathway_results"
TARGET_CLASS = "Alkaloids"
N_NETWORKS = 51

# -------------------------
# Loading spectra
# -------------------------
spectra_list = list(load_from_mgf(MGF_FILE))
print(f"Loaded {len(spectra_list)} spectra from MGF.")

for idx, s in enumerate(spectra_list):
    if s.get("feature_id") is None:
        s.set("feature_id", idx + 1)

# -------------------------
# Harmonizing
# -------------------------
spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS)
cleaned_spectra, report = spectrum_processor.process_spectra(
    spectra_list,
    cleaned_spectra_file=CLEANED_SPECTRA_FILE,
    create_report=True,
)
print("Spectra left after harmonisation:", len(cleaned_spectra))

# -------------------------
# Computing scores
# -------------------------
flash_modcosine_similarity = FlashSimilarity(
    score_type="cosine", matching_mode="hybrid", tolerance=0.05
)
flash_modcosine_scores = calculate_scores(
    cleaned_spectra,
    cleaned_spectra,
    similarity_function=flash_modcosine_similarity,
)
flash_modcosine_scores.to_json(SCORES_FILE)

# -------------------------
# Latin hypercube sampling
# -------------------------
settings = {
    "max_comp_size": [50, 300],
    "max_links": [5, 50],
    "cut_off": [0.6, 0.80],
}
param_sets, unit_samples = get_latin_hypercube_samples(
    settings, num_samples=N_NETWORKS, seed=27
)
discrepancy = qmc.discrepancy(unit_samples)
print("Discrepancy:", discrepancy)

with open(LHS_FILE, "w") as fout:
    json.dump(param_sets, fout)

# -------------------------
# Helper functions
# -------------------------


def topology_metrics(G):
    return {
        "avg_degree": round(calculate_average_degree(G), 2),
        "num_isolated_nodes": int(calculate_isolated_nodes(G)),
        "network_component_size_metric": round(
            network_component_size_metric(G, threshold=0.2), 2
        ),
    }


def compute_chemistry_metrics(df, G, key="component", attribute=TARGET_CHEM_LEVEL):
    net_avg_intra, net_avg_inter = calculate_intra_inter_similarity(df, key)
    return {
        "net_avg_intra": net_avg_intra,
        "net_avg_inter": net_avg_inter,
        "edge_purity": calculate_edge_purity(G, attribute=attribute),
        "component_purity": calculate_component_purity(G, key=key, attribute=attribute),
        "network_accuracy_score": calculate_network_accuracy_score(G),
        "consistency_measurement": calculate_consistency_measurement(
            G, key=key, attribute=attribute
        ),
    }


def compute_target_class_metrics(
    G,
    component_key="component",
    class_attr=TARGET_CHEM_LEVEL,
    target_class=TARGET_CLASS,
):
    return {
        "target_class": target_class,
        "edge_purity_target_incident": calculate_edge_purity_target_incident(
            G,
            attribute=class_attr,
            target_class=target_class,
            require_both_labeled=True,
        ),
        "component_purity_target_components": (
            calculate_component_purity_target_components(
                G,
                component_key=component_key,
                class_attr=class_attr,
                target_class=target_class,
                ignore_unlabeled=True,
                weight_by_target_nodes=True,
                min_component_size=2,
            )
        ),
        "target_component_purity": calculate_target_component_purity(
            G,
            component_key=component_key,
            class_attr=class_attr,
            target_class=target_class,
            ignore_unlabeled=True,
            min_component_size=2,
        ),
    }


def compute_networks(
    scores,
    score_name,
    max_comp_size,
    max_links,
    cut_off,
    similars_idx,
    similars_scores,
    identifier_key="feature_id",
):
    network = SimilarityNetworkMod(
        identifier_key=identifier_key,
        score_cutoff=cut_off,
        max_links=max_links,
        min_peaks=None,
        link_method="single",
        top_n=50,
    )
    network.create_network(
        scores,
        score_name=score_name,
        similars_idx=similars_idx,
        similars_scores=similars_scores,
    )
    network.filter_components(max_comp_size, cosine_delta=0.05)
    return network.graph, network.to_dataframe(col_name=identifier_key)


def safe_smiles_to_fp(smi):
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return smiles_to_morgan_fps(smi)
    except Exception:
        return None


# -------------------------
# Precompute top hits once
# -------------------------
score_name = flash_modcosine_scores.scores.data.dtype.names[0]
print("Precomputing top hits...")
_similars_idx, _similars_scores = get_top_hits(
    flash_modcosine_scores,
    identifier_key="feature_id",
    top_n=50,
    search_by="queries",
    score_name=score_name,
    ignore_diagonal=True,
)
print("Done.")

# -------------------------
# Load chemical info
# -------------------------
df_chem_info = pd.read_csv(CHEM_FILE)
df_chem_info["feature_id"] = pd.to_numeric(df_chem_info["feature_id"], errors="coerce")

# -------------------------
# Compute networks
# -------------------------
results = []

for idx, combinations in enumerate(param_sets):
    max_comp_size = combinations["max_comp_size"]
    max_links = combinations["max_links"]
    cut_off = combinations["cut_off"]

    net, df = compute_networks(
        scores=flash_modcosine_scores,
        score_name=score_name,
        max_comp_size=max_comp_size,
        max_links=max_links,
        cut_off=cut_off,
        similars_idx=_similars_idx,
        similars_scores=_similars_scores,
        identifier_key="feature_id",
    )
    print(f"Network {idx}/{len(param_sets)} computed")
    print(
        "nodes:",
        net.number_of_nodes(),
        "edges:",
        net.number_of_edges(),
        "avg_degree:",
        calculate_average_degree(net),
    )

    topology_net = topology_metrics(net)

    # Merge chem info
    df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
    df_chem_info_net = df_chem_info.merge(df, on="feature_id", how="inner")

    if df_chem_info_net.empty:
        results.append(
            {
                "params": combinations,
                "error": (
                    "merge empty (no matching feature_id between chem info and "
                    "network df)"
                ),
            }
        )
        print(f"Completed (merge empty) with parameters: {idx, combinations}")
        continue

    df_chem_info_net["fingerprint"] = df_chem_info_net["smiles"].apply(
        safe_smiles_to_fp
    )

    prepare_graph_class(
        net, df_chem_info_net, feature_col="feature_id", attribute=TARGET_CHEM_LEVEL
    )
    prepare_graph_fps(
        net, df_chem_info_net, feature_col="feature_id", attribute="fingerprint"
    )

    chemistry_metrics = compute_chemistry_metrics(
        df_chem_info_net, net, key="component"
    )

    results.append(
        {
            "params": combinations,
            "topology_metrics": topology_net,
            "chemistry_metrics": chemistry_metrics,
        }
    )

    print(f"Completed with parameters: {combinations}")

# -------------------------
# Save output
# -------------------------
with open(NETWORKS_OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

# -------------------------
# Optimization
# -------------------------


def make_df_for_score(score_name: str, entries) -> pd.DataFrame:
    """Flatten entries for a given score into a DataFrame with top_/chem_ prefixes."""
    rows = []
    for entry in entries:
        row = {}
        for section, section_dict in entry.items():
            if not isinstance(section_dict, dict):
                continue
            if section == "topology_metrics":
                prefix = "top_"
            elif section == "chemistry_metrics":
                prefix = "chem_"
            else:
                prefix = ""
            for k, v in section_dict.items():
                row[f"{prefix}{k}"] = v
        row["score_family"] = score_name
        rows.append(row)
    return pd.DataFrame(rows)


df = make_df_for_score("Modified_Cosine", results)
print(df.head())

maximize_user = [
    "top_network_component_size_metric",
    "top_avg_degree",
    "chem_net_avg_intra",
    "chem_consistency_measurement",
]
minimize_user = [
    "top_num_isolated_nodes",
]

all_cols = df.columns.tolist()
metric_cols = [c for c in all_cols if c.startswith(("top_", "chem_"))]
param_cols = [
    c
    for c in all_cols
    if c not in metric_cols + ["score_family", "composite_score", "is_pareto"]
]


def split_metrics(metric_cols, maximize_hint=None, minimize_hint=None):
    if maximize_hint or minimize_hint:
        max_cols = [c for c in (maximize_hint or []) if c in metric_cols]
        min_cols = [c for c in (minimize_hint or []) if c in metric_cols]
        return max_cols, min_cols

    max_like = ("avg", "network_component_size_metric", "intra", "accuracy")
    min_like = ("isolated",)

    max_cols, min_cols = [], []
    for c in metric_cols:
        name = c.lower()
        if any(k in name for k in min_like):
            min_cols.append(c)
        elif any(k in name for k in max_like):
            max_cols.append(c)
        else:
            max_cols.append(c)
    return max_cols, min_cols


maximize, minimize = split_metrics(
    metric_cols,
    maximize_hint=maximize_user,
    minimize_hint=minimize_user,
)
print("Maximize:", maximize)
print("Minimize:", minimize)

weights = {}
df["composite_score"] = np.nan

for fam in df["score_family"].unique():
    sub = df[df.score_family == fam]
    Zmax = (
        StandardScaler().fit_transform(sub[maximize])
        if maximize
        else np.zeros((len(sub), 0))
    )
    Zmin = (
        -StandardScaler().fit_transform(sub[minimize])
        if minimize
        else np.zeros((len(sub), 0))
    )
    w_max = np.array([weights.get(c, 1.0) for c in maximize])
    w_min = np.array([weights.get(c, 1.0) for c in minimize])
    score = (Zmax @ w_max if w_max.size else 0) + (Zmin @ w_min if w_min.size else 0)
    df.loc[sub.index, "composite_score"] = score

best_overall = (
    df.sort_values(["score_family", "composite_score"], ascending=[True, False])
    .groupby("score_family", as_index=False)
    .head(10)
)
best_overall.to_csv(TOP_CONFIGS_FILE, index=False)

# -------------------------
# Build top 3 networks
# -------------------------
for rank, (i, row) in enumerate(best_overall.iloc[0:3].iterrows(), start=1):
    network = SimilarityNetworkMod(
        identifier_key="feature_id",
        score_cutoff=row["cut_off"],
        max_links=int(row["max_links"]),
        min_peaks=None,
        link_method="single",
        top_n=50,
    )
    network.create_network(
        flash_modcosine_scores,
        score_name=score_name,
        similars_idx=_similars_idx,
        similars_scores=_similars_scores,
    )
    network.filter_components(int(row["max_comp_size"]), cosine_delta=0.05)
    network.export_to_file(f"top{rank}_graph.graphml", graph_format="graphml")

print(f"Total runtime: {time.time()-t0:.1f}s")
