"""Unified arteMIS network-optimisation pipeline (global / seed / target modes).

Replaces the three near-duplicate example scripts: the only real
differences between modes are which chemistry metric is computed
(artemis.pipeline.metrics) and how the chemical-info table is filtered
before merging into each network (_filter_chem_info below).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

from matchms import calculate_scores
from matchms.filtering.default_pipelines import CLEAN_PEAKS, DEFAULT_FILTERS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.importing import load_from_mgf
from matchms.networking.networking_functions import get_top_hits
from matchms.similarity.FlashSimilarity import FlashCosine

from artemis.networking.SimilarityNetworkMod import SimilarityNetworkMod
from artemis.pipeline.config import PipelineConfig
from artemis.pipeline.metrics import compute_chemistry_metrics, compute_topology_metrics
from artemis.utils.fps import smiles_to_morgan_fps
from artemis.utils.lhs import get_latin_hypercube_samples
from artemis.utils.prepare_graph import prepare_graph_class, prepare_graph_fps

# Similarity-function builders, keyed by the `scores:` keys used in config.yaml.
# `matching_mode` defines what the score family *is*, so it stays fixed here;
# `tolerance` is a tunable and comes from config.yaml (scores.<key>.tolerance).
SCORE_FUNCTIONS = {
    "modcosine": lambda tolerance: FlashCosine(
        matching_mode="hybrid", tolerance=tolerance
    ),
    "cosine": lambda tolerance: FlashCosine(
        matching_mode="fragment", tolerance=tolerance
    ),
}


def safe_smiles_to_fp(smi):
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        return smiles_to_morgan_fps(smi)
    except Exception:
        return None


def _precompute_top_hits(scores_obj, ids, top_n):
    # Support both new matchms API (score_fields) and old API (scores.data.dtype.names)
    if hasattr(scores_obj, "score_fields"):
        sn = scores_obj.score_fields[0]
        sim_idx, sim_scores = get_top_hits(
            scores_obj,
            top_n=top_n,
            axis=1,
            score_name=sn,
            identifiers=ids,
            ignore_diagonal=True,
        )
    else:
        sn = scores_obj.scores.data.dtype.names[0]
        sim_idx, sim_scores = get_top_hits(
            scores_obj,
            identifier_key="feature_id",
            top_n=top_n,
            search_by="queries",
            score_name=sn,
            ignore_diagonal=True,
        )
    return sn, sim_idx, sim_scores


def _build_network(params, score_cfg):
    # top_n only needs to satisfy `top_n >= max_links` (see SimilarityNetworkMod.create_network);
    # the actual neighbor candidates come from the precomputed similars_idx/similars_scores.
    network = SimilarityNetworkMod(
        score_cutoff=params["cut_off"],
        max_links=int(params["max_links"]),
        min_peaks=int(params["matching_peaks"])
        if score_cfg["use_matching_peaks"] and "matching_peaks" in params
        else None,
        link_method="single",
        top_n=int(params["max_links"]) + 1,
    )
    network.create_network(
        score_cfg["scores"],
        score_name=score_cfg["score_name"],
        similars_idx=score_cfg["similars_idx"],
        similars_scores=score_cfg["similars_scores"],
        identifiers=score_cfg["ids"],
    )
    network.filter_components(int(params["max_comp_size"]), cosine_delta=0.05)
    return network


def _filter_chem_info(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    col = cfg.chem_level
    if cfg.mode == "target":
        return df[df[col].notna() & (df[col] != "")].copy().reset_index(drop=True)
    if cfg.mode == "seed":
        return (
            df[df[col].notna() & (df[col] != cfg.unlabeled_value)]
            .copy()
            .reset_index(drop=True)
        )
    return df.copy()  # global: use every annotated-or-not row


def _make_df_for_score(score_family: str, entries) -> pd.DataFrame:
    prefix_map = {"topology_metrics": "top_", "chemistry_metrics": "chem_"}
    rows = []
    for entry in entries:
        row = {
            f"{prefix_map.get(section, '')}{k}": v
            for section, section_dict in entry.items()
            if isinstance(section_dict, dict)
            for k, v in section_dict.items()
        }
        row["score_family"] = score_family
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: PipelineConfig) -> None:
    t0 = time.time()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"Mode: {cfg.mode} | Scores: {list(cfg.scores)}")

    cleaned_spectra_file = os.path.join(cfg.out_dir, "cleaned_spectra.mgf")
    lhs_file = os.path.join(cfg.out_dir, "LHS.json")
    networks_file = os.path.join(cfg.out_dir, "networks.json")
    top_configs_file = os.path.join(cfg.out_dir, "top_configs.csv")

    # -------------------------
    # Load and harmonize spectra
    # -------------------------
    spectra_list = list(load_from_mgf(cfg.mgf_file))
    print(f"Loaded {len(spectra_list)} spectra from MGF.")
    for idx, s in enumerate(spectra_list):
        if s.get("feature_id") is None:
            s.set("feature_id", idx + 1)

    filters = DEFAULT_FILTERS + CLEAN_PEAKS if cfg.clean_peaks else DEFAULT_FILTERS
    cleaned_spectra, _ = SpectrumProcessor(filters).process_spectra(
        spectra_list,
        cleaned_spectra_file=cleaned_spectra_file,
        create_report=True,
    )
    print("Spectra left after harmonisation:", len(cleaned_spectra))
    feature_ids = [s.get("feature_id") for s in cleaned_spectra]

    # -------------------------
    # Compute similarity scores (one per requested score family)
    # -------------------------
    score_configs = {}
    for label in cfg.scores:
        if label not in SCORE_FUNCTIONS:
            raise ValueError(
                f"Unknown score family {label!r}. Known: {list(SCORE_FUNCTIONS)}"
            )
        tolerance = cfg.scores[label]["tolerance"]
        scores_obj = calculate_scores(
            cleaned_spectra,
            cleaned_spectra,
            similarity_function=SCORE_FUNCTIONS[label](tolerance),
        )
        scores_obj.save(os.path.join(cfg.out_dir, f"{label}-scores.npz"))
        score_configs[label] = {
            "scores": scores_obj,
            "ids": feature_ids,
            "use_matching_peaks": True,
        }

    # -------------------------
    # Latin hypercube sampling
    # -------------------------
    param_sets, unit_samples = get_latin_hypercube_samples(
        cfg.param_space.bounds,
        num_samples=cfg.param_space.n_networks,
        seed=cfg.param_space.seed,
    )
    with open(lhs_file, "w") as f:
        json.dump(param_sets, f)

    # -------------------------
    # Precompute neighbor candidates once per score family, wide enough for
    # the largest max_links sampled (top_n per network is always max_links + 1)
    # -------------------------
    precompute_top_n = cfg.param_space.max_links_upper_bound + 1
    print("Precomputing top hits for all scores...")
    for label, sc in score_configs.items():
        print(f"  {label}...")
        sn, sim_idx, sim_scores = _precompute_top_hits(
            sc["scores"], sc["ids"], precompute_top_n
        )
        sc["score_name"], sc["similars_idx"], sc["similars_scores"] = (
            sn,
            sim_idx,
            sim_scores,
        )
    print("Done.")

    # -------------------------
    # Load chemical info
    # -------------------------
    df_chem_info = pd.read_csv(cfg.chem_file)
    df_chem_info["feature_id"] = pd.to_numeric(
        df_chem_info["feature_id"], errors="coerce"
    )
    if cfg.chem_level not in df_chem_info.columns:
        raise ValueError(
            f"chem_level column {cfg.chem_level!r} not found in chem_file {cfg.chem_file!r}. "
            f"Available columns: {list(df_chem_info.columns)}"
        )
    df_chem_info = _filter_chem_info(df_chem_info, cfg)
    df_chem_info["fingerprint"] = df_chem_info["smiles"].apply(safe_smiles_to_fp)

    # -------------------------
    # Build networks + compute metrics
    # -------------------------
    all_results = {}
    for score_label, score_cfg in score_configs.items():
        print(f"\n--- Building networks for: {score_label} ---")
        results = []
        for idx, params in enumerate(param_sets):
            network = _build_network(params, score_cfg)
            net = network.graph
            df = network.to_dataframe(col_name="feature_id")
            df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
            print(f"[{score_label}] Network {idx}/{len(param_sets)} computed")

            df_merged = df_chem_info.merge(df, on="feature_id", how="inner")
            entry = {
                "params": params,
                "topology_metrics": compute_topology_metrics(net),
            }

            if df_merged.empty:
                entry["error"] = "merge empty (no matching feature_id)"
                print(f"[{score_label}] Merge empty for params: {params}")
                results.append(entry)
                continue

            prepare_graph_class(
                net, df_merged, feature_col="feature_id", attribute=cfg.chem_level
            )
            prepare_graph_fps(
                net, df_merged, feature_col="feature_id", attribute="fingerprint"
            )

            chem_metrics = compute_chemistry_metrics(
                cfg.mode,
                df_merged,
                net,
                key="component",
                attribute=cfg.chem_level,
                target_class=cfg.target_class,
            )
            if cfg.mode == "target":
                chem_metrics[
                    "target_class"
                ] = cfg.target_class  # descriptive only, not optimizable
            entry["chemistry_metrics"] = chem_metrics
            results.append(entry)
            print(f"[{score_label}] Completed with parameters: {params}")

        all_results[score_label] = results

    with open(networks_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # -------------------------
    # Optimization (per score family)
    # Note: score_family is kept as the raw config key (e.g. "modcosine"), not
    # the human-readable label, so it round-trips through the top-3 export below.
    # -------------------------
    df = pd.concat(
        [_make_df_for_score(label, results) for label, results in all_results.items()],
        ignore_index=True,
    )

    maximize = cfg.optimization.maximize
    minimize = cfg.optimization.minimize
    print("Maximize:", maximize)
    print("Minimize:", minimize)

    df = df.dropna(subset=maximize + minimize).reset_index(drop=True)

    weights = cfg.optimization.weights
    w_max = np.array([weights.get(c, 1.0) for c in maximize])
    w_min = np.array([weights.get(c, 1.0) for c in minimize])

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
        score = (Zmax @ w_max if w_max.size else 0) + (
            Zmin @ w_min if w_min.size else 0
        )
        df.loc[sub.index, "composite_score"] = score

    best_overall = (
        df.sort_values(["score_family", "composite_score"], ascending=[True, False])
        .groupby("score_family", as_index=False)
        .head(10)
    )
    best_overall.to_csv(top_configs_file, index=False)

    for score_label, score_cfg in score_configs.items():
        score_best = best_overall[best_overall["score_family"] == score_label]
        for rank, (_, row) in enumerate(score_best.head(3).iterrows(), start=1):
            _build_network(row, score_cfg).export_to_file(
                os.path.join(cfg.out_dir, f"{score_label}_top{rank}_graph.graphml"),
                graph_format="graphml",
            )

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("Discrepancy:", qmc.discrepancy(unit_samples))
