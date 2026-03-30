# General Importing 
import sys, os
from scipy.stats import qmc
import json
import pandas as pd

# Loading spectra
from matchms.importing import load_from_mgf

spectra_list = list(load_from_mgf("path_to_your_file.mgf"))
print(f"Loaded {len(spectra_list)} spectra from MGF.")

for idx, s in enumerate(spectra_list):
    if s.get('feature_id') == None:
        s.set("feature_id", idx + 1)

# Harmonizing, without filtering
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from matchms.exporting import save_as_mgf

spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS)

final_filter_order = [filter.__name__ for filter in spectrum_processor.filters]

cleaned_spectra, report = spectrum_processor.process_spectra(spectra_list, cleaned_spectra_file = "name_of_your_cleaned_file.mgf",create_report=True)

print("Spectra left after harmonzation:", len(cleaned_spectra))

# Computing scores
from matchms.similarity.FlashSimilarity import FlashSimilarity
from matchms import calculate_scores

flash_modcosine_similarity = FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=0.05)
flash_modcosine_scores = calculate_scores(cleaned_spectra, cleaned_spectra, similarity_function=flash_modcosine_similarity)
flash_modcosine_scores.to_json("NAICONS_101_CS_OUT_SCORES.json")

# Latin hyper cube
from artemis.utils.lhs import get_latin_hypercube_samples

n = 51 # 50 networks
# compute n networks
settings = {"max_comp_size": [50,300], "max_links": [5,50], "cut_off": [0.6,0.80]}

param_sets, unit_samples = get_latin_hypercube_samples(settings, num_samples=n, seed=27)

discrepancy = qmc.discrepancy(unit_samples)
print("Discrepancy:", discrepancy)

with open('NAICONS_101_CS_LHS_SETTINGS.json', 'w') as fout:
    json.dump(param_sets, fout)

############
# Compute networks
# Networking Importing
from artemis.networking.SimilarityNetworkMod import SimilarityNetworkMod
from artemis.utils.fps import smiles_to_morgan_fps
from artemis.utils.prepare_graph import prepare_graph_fps, prepare_graph_class

# Evaluation Importing
from artemis.evaluation.topology_metrics import (
    calculate_average_degree,
    calculate_isolated_nodes,
    network_component_size_metric,
)
from artemis.evaluation.chemistry_metrics import (
    calculate_intra_inter_similarity,
    calculate_edge_purity,
    calculate_component_purity,
    calculate_network_accuracy_score,
    calculate_consistency_measurement,
    calculate_edge_purity_target_incident,
    calculate_component_purity_target_components,
    calculate_target_component_purity
)
# -------------------------
# Helper functions
# -------------------------
target_chem_level = "npc_pathway_results" #to change
target_class = "Alkaloids"   # not used here

def topology_metrics(G):
    return {
        "avg_degree": round(calculate_average_degree(G), 2),
        "num_isolated_nodes": int(calculate_isolated_nodes(G)),
        "network_component_size_metric": round(network_component_size_metric(G, threshold=0.2), 2),
    }

def compute_chemistry_metrics(df, G, key="component", attribute=target_chem_level):
    net_avg_intra, net_avg_inter = calculate_intra_inter_similarity(df, key)
    return {
        "net_avg_intra": net_avg_intra,
        "net_avg_inter": net_avg_inter,
        "edge_purity": calculate_edge_purity(G, attribute=attribute),
        "component_purity": calculate_component_purity(G, key=key, attribute=attribute),
        "network_accuracy_score": calculate_network_accuracy_score(G),
        "consistency_measurement": calculate_consistency_measurement(G, key=key, attribute=attribute),
    }

def compute_target_class_metrics(G, component_key="component", class_attr=target_chem_level, target_class=target_class):
    return {
        "target_class": target_class,
        "edge_purity_target_incident": calculate_edge_purity_target_incident(G, attribute=class_attr, target_class=target_class, require_both_labeled=True),
        "component_purity_target_components": calculate_component_purity_target_components(G, component_key=component_key, class_attr=class_attr, target_class=target_class, ignore_unlabeled=True, weight_by_target_nodes=True, min_component_size=2),
        "target_component_purity": calculate_target_component_purity(G, component_key=component_key, class_attr=class_attr, target_class=target_class, ignore_unlabeled=True, min_component_size=2),
    }

def compute_networks(scores, score_name, max_comp_size, max_links, cut_off, identifier_key="feature_id"):
    network = SimilarityNetworkMod(
        identifier_key=identifier_key,
        score_cutoff=cut_off,
        max_links=max_links,
        min_peaks=None,
        link_method="single",
        top_n=50,
    )
    network.create_network(scores, score_name=score_name)
    network.filter_components(max_comp_size, cosine_delta=0.05)
    G = network.graph
    df_net = network.to_dataframe(col_name=identifier_key)
    return G, df_net

def safe_smiles_to_fp(smi):
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return smiles_to_morgan_fps(smi)
    except Exception:
        return None
    
score_name = flash_modcosine_scores.scores.data.dtype.names[0]

df_chem_info = pd.read_csv("chemical_annotations".csv")  # <-- put your MS2Query results file
df_chem_info["feature_id"] = pd.to_numeric(df_chem_info["feature_id"], errors="coerce")

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
        identifier_key="feature_id",
    )

    print("Network computed")
    print(
    "nodes:", net.number_of_nodes(),
    "edges:", net.number_of_edges(),
    "avg_degree:", calculate_average_degree(net)
    )

    topology_net = topology_metrics(net)

    # Merge chem info
    df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
    df_chem_info_net = df_chem_info.merge(df, on="feature_id", how="inner")
    # If merge empty, store and continue
    if df_chem_info_net.empty:
        results.append({
            "params": combinations,
            "error": "merge empty (no matching feature_id between chem info and network df, most likely ms2query results and harmonized file have different dimentions)"
        })
        print(f"Completed (merge empty) with parameters: {idx, combinations}")
        continue

    # Fingerprints safely
    df_chem_info_net["fingerprint"] = df_chem_info_net["smiles"].apply(safe_smiles_to_fp)

    # Add attributes to graph
    prepare_graph_class(net, df_chem_info_net, feature_col="feature_id", attribute=target_chem_level)
    prepare_graph_fps(net, df_chem_info_net, feature_col="feature_id", attribute="fingerprint")

    chemistry_metrics = compute_chemistry_metrics(df_chem_info_net, net, key="component")
    #target_metrics = compute_target_class_metrics(net, component_key="component", class_attr=target_chem_level, target_class=target_class)

    results.append({
        "params": combinations,
        "topology_metrics": topology_net,
        "chemistry_metrics": chemistry_metrics,
        #"target_class_metrics": target_metrics,
    })

    print(f"Completed with parameters: {combinations}")

# -------------------------
# Save output
# -------------------------
with open("evaluation_network_results_.json", "w") as f:
    json.dump(results, f, indent=4)


### Optimization
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd

# Helper function
def make_df_for_score(score_name: str, entries) -> pd.DataFrame:
    """Flatten entries for a given score into a DataFrame with top_/chem_ prefixes."""
    rows = []
    for entry in entries:
        row = {}
        # prefix by type
        for section, section_dict in entry.items():
            if not isinstance(section_dict, dict):
                continue
            if section == "topology_metrics":
                prefix = "top_"
            elif section == "chemistry_metrics":
                prefix = "chem_"
            else:
                prefix = ""  # no prefix for params
            for k, v in section_dict.items():
                row[f"{prefix}{k}"] = v
        row["score_family"] = score_name
        rows.append(row)
    return pd.DataFrame(rows)


df = make_df_for_score("Modified_Cosine", results)
print(df.head())

# Parameter columns (as they appear in your JSON)
param_cols = ["max_comp_size", "max_links", "cut_off"]

#chose
maximize_user = [
    "top_network_component_size_metric",
    "top_avg_degree",
    "chem_net_avg_intra",
    #"chem_net_avg_inter",
    #"chem_edge_purity",
    #"chem_component_purity",
    #"chem_network_accuracy_score",
    "chem_consistency_measurement",
]
minimize_user = [
    "top_num_isolated_nodes",
]


# ------------- DETECT METRICS & PARAMS -------------
all_cols = df.columns.tolist()

# metrics start with 'top_' or 'chem_'
metric_cols = [c for c in all_cols if c.startswith(("top_", "chem_"))]

# everything else (except score_family) are params
param_cols = [c for c in all_cols if c not in metric_cols + ["score_family", "composite_score", "is_pareto"]]


def split_metrics(metric_cols, maximize_hint=None, minimize_hint=None):
    if maximize_hint or minimize_hint:
        max_cols = [c for c in (maximize_hint or []) if c in metric_cols]
        min_cols = [c for c in (minimize_hint or []) if c in metric_cols]
        return max_cols, min_cols

    max_like = ("avg", "network_component_size_metric", "intra", "accuracy")
    min_like = ("isolated",)  # <-- comma is essential

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

maximize, minimize = split_metrics(metric_cols, 
    maximize_hint=maximize_user, 
    minimize_hint=minimize_user)

print("Maximize:", maximize)
print("Minimize:", minimize)

# ------------- COMPOSITE SCORE -------------
weights = {}  # optional weighting per metric, e.g. {"chem_net_avg_intra": 2.0}

df["composite_score"] = np.nan

for fam in df["score_family"].unique():
    sub = df[df.score_family == fam]
    Zmax = StandardScaler().fit_transform(sub[maximize]) if maximize else np.zeros((len(sub), 0))
    Zmin = -StandardScaler().fit_transform(sub[minimize]) if minimize else np.zeros((len(sub), 0))
    w_max = np.array([weights.get(c, 1.0) for c in maximize])
    w_min = np.array([weights.get(c, 1.0) for c in minimize])
    score = (Zmax @ w_max if w_max.size else 0) + (Zmin @ w_min if w_min.size else 0)
    df.loc[sub.index, "composite_score"] = score


# ------------- EXPORT -------------
best_overall = (
    df.sort_values(["score_family", "composite_score"], ascending=[True, False])
          .groupby("score_family", as_index=False)
          .head(10)
)

best_overall.to_csv("top_configurations.csv", index=False)

# Build the top 3 networks
from artemis.networking.build import build_similarity_graph

for rank, (i, row) in enumerate(best_overall.iloc[0:3].iterrows(), start=1):
    graph = build_similarity_graph(
        flash_modcosine_scores,
        score_name=score_name,
        identifier_key="feature_id",
        cut_off=row['cut_off'],
        max_links=int(row['max_links']),
        top_n=50,
        max_comp_size=int(row['max_comp_size']),
        link_method="single",
        min_peaks=None,
    )
    graph.export_to_file(f"top{rank}_graph.graphml", graph_format="graphml")

