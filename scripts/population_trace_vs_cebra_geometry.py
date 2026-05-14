#!/usr/bin/env python3
"""
Population-averaged calcium summaries vs CEBRA latent task geometry.

This analysis asks whether conserved CEBRA geometry is explained by simple
gross calcium dynamics. It intentionally does NOT compare raw neuron vectors
across animals, because neurons are not matched across animals.

Raw calcium is reduced to neuron-identity-free summaries:
  - population-averaged calcium trace over CSUS5 task bins
  - fraction-active/recruited trace over CSUS5 task bins
  - distribution of preferred task bins and modulation amplitudes

CEBRA is analyzed as relational task geometry:
  - task-bin latent trajectory
  - task-bin distance matrix
  - cross-animal similarity of vectorized task-state distances
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import ks_2samp, pearsonr, ttest_rel, wasserstein_distance, wilcoxon
from sklearn.preprocessing import StandardScaler

from raw_vs_cebra_geometry_comparison import (
    GeometryOptions,
    SessionSpec,
    extract_task_aligned_calcium,
    get_animal_data,
    get_calcium_session,
    load_cebra_npz,
    load_sessions_csv,
    make_task_bin_labels,
)


@dataclass
class PopulationGeometryOptions:
    task_scheme: str = "CSUS5"
    win: Tuple[float, float] = (0.0, 2.0)
    pre_cs_win: Tuple[float, float] = (-1.0, 0.0)
    csus5_edges: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    n_shuffles: int = 500
    random_seed: int = 1
    min_trials: int = 5
    min_neurons: int = 10
    activity_threshold_z: float = 1.0
    output_dir: Path = Path("population_trace_vs_cebra_geometry_output")
    cebra_embedding_key: Optional[str] = None
    cebra_label_key: Optional[str] = None
    cebra_bin_vectors_key: Optional[str] = None
    procrustes_cebra: bool = True


def run_population_trace_vs_cebra_geometry(
    sessions: Sequence[SessionSpec],
    opts: Optional[PopulationGeometryOptions] = None,
) -> Dict[str, Any]:
    opts = opts or PopulationGeometryOptions()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.random_seed)

    print("\nPopulation trace vs CEBRA latent task geometry")
    print(f"  Sessions listed: {len(sessions)}")
    print(f"  Output dir: {opts.output_dir}")

    raw_records = []
    cebra_records = []
    for idx, session in enumerate(sessions, start=1):
        print(f"  [{idx}/{len(sessions)}] {session.rat_name} {session.date_str}")
        try:
            raw_records.append(compute_raw_population_record(session, opts))
        except Exception as exc:
            warnings.warn(f"{session.key}: raw population summary failed ({exc}); skipping.")
        try:
            cebra_records.append(compute_cebra_geometry_record(session, opts))
        except Exception as exc:
            warnings.warn(f"{session.key}: CEBRA geometry failed ({exc}); skipping.")

    shared = sorted(set(r["name"] for r in raw_records) & set(r["name"] for r in cebra_records))
    raw_records = sorted([r for r in raw_records if r["name"] in shared], key=lambda r: r["name"])
    cebra_records = sorted([r for r in cebra_records if r["name"] in shared], key=lambda r: r["name"])

    raw_trace = compare_population_traces(raw_records, opts, rng)
    distributional = compare_distributional_modulation(raw_records)
    cebra_geometry = compare_cebra_geometry(cebra_records, opts, rng)
    paired_stats = compare_raw_trace_to_cebra_geometry(raw_trace, cebra_geometry)

    results = {
        "opts": opts,
        "raw_records": raw_records,
        "cebra_records": cebra_records,
        "raw_trace": raw_trace,
        "distributional": distributional,
        "cebra_geometry": cebra_geometry,
        "paired_stats": paired_stats,
    }
    save_outputs(results, opts)
    plot_population_trace_vs_cebra_geometry(results, opts)

    print(f"  Raw valid datasets: {len(raw_records)}")
    print(f"  CEBRA valid datasets: {len(cebra_records)}")
    print(f"  Animal pairs: {len(raw_trace['pairs'])}")
    print(f"\nDone. Results saved to {opts.output_dir}")
    return results


def helper_geometry_opts(opts: PopulationGeometryOptions) -> GeometryOptions:
    return GeometryOptions(
        task_schemes=(opts.task_scheme,),
        win=opts.win,
        pre_cs_win=opts.pre_cs_win,
        csus5_edges=opts.csus5_edges,
        min_trials=opts.min_trials,
        min_neurons=opts.min_neurons,
        cebra_embedding_key=opts.cebra_embedding_key,
        cebra_label_key=opts.cebra_label_key,
        cebra_bin_vectors_key=opts.cebra_bin_vectors_key,
    )


def compute_raw_population_record(session: SessionSpec, opts: PopulationGeometryOptions) -> Dict[str, Any]:
    """Collapse raw calcium into neuron-identity-free population summaries."""
    animal = get_animal_data(session, None)
    calcium_all, calcium_ts, trial_cs = get_calcium_session(animal, session)
    if len(trial_cs) < opts.min_trials:
        raise ValueError(f"only {len(trial_cs)} CS trials")

    align_win = (min(opts.win[0], opts.pre_cs_win[0]), max(opts.win[1], opts.pre_cs_win[1]))
    aligned, aligned_ts = extract_task_aligned_calcium(calcium_all, calcium_ts, trial_cs, align_win)
    if aligned.shape[2] < opts.min_neurons:
        raise ValueError(f"only {aligned.shape[2]} neurons")

    helper_opts = helper_geometry_opts(opts)
    bin_labels, bin_edges, bin_names = make_task_bin_labels(aligned_ts, opts.task_scheme, helper_opts)
    task_time_mask = np.isfinite(bin_labels)
    z_aligned = zscore_neurons_over_task_samples(aligned, task_time_mask)

    mean_trace_time = np.nanmean(aligned[:, task_time_mask, :], axis=(0, 2))
    mean_trace_time_z = zscore_1d(mean_trace_time)
    fraction_active_time = np.nanmean(z_aligned[:, task_time_mask, :] > opts.activity_threshold_z, axis=(0, 2))
    fraction_active_time_z = zscore_1d(fraction_active_time)

    mean_trace_bins = bin_trace_from_aligned(aligned, bin_labels, reducer="mean")
    mean_trace_bins_z = zscore_1d(mean_trace_bins)
    fraction_active_bins = fraction_active_trace(z_aligned, bin_labels, opts.activity_threshold_z)
    fraction_active_bins_z = zscore_1d(fraction_active_bins)

    neuron_profiles = neuron_task_profiles(z_aligned, bin_labels)
    preferred_bins = np.nanargmax(neuron_profiles, axis=1) + 1
    modulation = np.nanmax(neuron_profiles, axis=1) - np.nanmean(neuron_profiles, axis=1)
    normalized_profiles = zscore_rows(neuron_profiles)
    preferred_hist = normalized_histogram(preferred_bins, n_bins=len(bin_edges) - 1)
    sorted_profiles = normalized_profiles[np.argsort(preferred_bins)]

    return {
        "name": session.key,
        "rat_name": session.rat_name,
        "date_str": session.date_str,
        "aligned_ts_task": aligned_ts[task_time_mask],
        "bin_edges": bin_edges,
        "bin_names": bin_names,
        "raw_mean_trace_time": mean_trace_time,
        "raw_mean_trace_time_z": mean_trace_time_z,
        "fraction_active_trace_time": fraction_active_time,
        "fraction_active_trace_time_z": fraction_active_time_z,
        "raw_mean_trace_bins": mean_trace_bins,
        "raw_mean_trace_bins_z": mean_trace_bins_z,
        "fraction_active_trace_bins": fraction_active_bins,
        "fraction_active_trace_bins_z": fraction_active_bins_z,
        "neuron_profiles": neuron_profiles,
        "preferred_bins": preferred_bins,
        "preferred_hist": preferred_hist,
        "modulation_amplitude": modulation,
        "normalized_profiles_sorted": sorted_profiles,
    }


def zscore_neurons_over_task_samples(aligned: np.ndarray, task_time_mask: np.ndarray) -> np.ndarray:
    out = aligned.copy()
    for neuron_idx in range(out.shape[2]):
        vals = out[:, task_time_mask, neuron_idx].reshape(-1)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        out[:, :, neuron_idx] = (out[:, :, neuron_idx] - mu) / sigma
    return out


def bin_trace_from_aligned(aligned: np.ndarray, bin_labels: np.ndarray, reducer: str = "mean") -> np.ndarray:
    n_bins = int(np.nanmax(bin_labels))
    trace = np.full(n_bins, np.nan)
    for bin_idx in range(1, n_bins + 1):
        mask = bin_labels == bin_idx
        if reducer == "mean":
            trace[bin_idx - 1] = np.nanmean(aligned[:, mask, :])
        else:
            raise ValueError(f"unknown reducer: {reducer}")
    return trace


def fraction_active_trace(z_aligned: np.ndarray, bin_labels: np.ndarray, threshold: float) -> np.ndarray:
    n_bins = int(np.nanmax(bin_labels))
    trace = np.full(n_bins, np.nan)
    active = z_aligned > threshold
    for bin_idx in range(1, n_bins + 1):
        mask = bin_labels == bin_idx
        trace[bin_idx - 1] = np.nanmean(active[:, mask, :])
    return trace


def neuron_task_profiles(z_aligned: np.ndarray, bin_labels: np.ndarray) -> np.ndarray:
    n_bins = int(np.nanmax(bin_labels))
    profiles = np.full((z_aligned.shape[2], n_bins), np.nan)
    for bin_idx in range(1, n_bins + 1):
        mask = bin_labels == bin_idx
        profiles[:, bin_idx - 1] = np.nanmean(z_aligned[:, mask, :], axis=(0, 1))
    return profiles


def zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sigma = np.nanstd(values)
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0
    return (values - mu) / sigma


def zscore_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values, axis=1, keepdims=True)
    sigma = np.nanstd(values, axis=1, keepdims=True)
    sigma[(~np.isfinite(sigma)) | (sigma == 0)] = 1.0
    return (values - mu) / sigma


def normalized_histogram(preferred_bins: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(preferred_bins.astype(int), minlength=n_bins + 1)[1 : n_bins + 1]
    total = np.sum(counts)
    return counts / total if total > 0 else np.full(n_bins, np.nan)


def compute_cebra_geometry_record(session: SessionSpec, opts: PopulationGeometryOptions) -> Dict[str, Any]:
    """Compute CEBRA task trajectory and task-state distance matrix."""
    if session.npz_path is None:
        raise ValueError("missing npz_path")
    helper_opts = helper_geometry_opts(opts)
    embedding, labels, embedding_key, label_key = load_cebra_npz(session.npz_path, helper_opts)

    pre_binned = embedding_key in {"zB_runs", "zA_runs", "bin_vectors", "binVectors"} or (
        embedding.shape[0] == len(labels) and len(labels) <= 20
    )
    if pre_binned:
        trajectory = embedding
        bin_labels = labels
    else:
        keep = labels != 0
        embedding = embedding[keep]
        labels = labels[keep]
        bin_labels = np.asarray(sorted(np.unique(labels)))
        trajectory = np.vstack([np.nanmean(embedding[labels == label], axis=0) for label in bin_labels])

    rdm = squareform(pdist(trajectory, metric="euclidean"))
    return {
        "name": session.key,
        "rat_name": session.rat_name,
        "date_str": session.date_str,
        "trajectory": trajectory,
        "bin_labels": bin_labels,
        "rdm": rdm,
        "rdm_vector": upper_triangle(rdm),
        "embedding_key": embedding_key,
        "label_key": label_key,
    }


def compare_population_traces(
    records: Sequence[Mapping[str, Any]], opts: PopulationGeometryOptions, rng: np.random.Generator
) -> Dict[str, Any]:
    pairs = []
    n = len(records)
    mean_sim = np.full((n, n), np.nan)
    frac_sim = np.full((n, n), np.nan)
    mean_dist = np.full((n, n), np.nan)
    frac_dist = np.full((n, n), np.nan)
    np.fill_diagonal(mean_sim, 1.0)
    np.fill_diagonal(frac_sim, 1.0)
    np.fill_diagonal(mean_dist, 0.0)
    np.fill_diagonal(frac_dist, 0.0)

    mean_shuffle = []
    frac_shuffle = []
    for i, a in enumerate(records):
        for j, b in enumerate(records):
            if j <= i:
                continue
            mean_a = np.asarray(a["raw_mean_trace_bins_z"])
            mean_b = np.asarray(b["raw_mean_trace_bins_z"])
            frac_a = np.asarray(a["fraction_active_trace_bins_z"])
            frac_b = np.asarray(b["fraction_active_trace_bins_z"])
            mean_r = safe_pearson(mean_a, mean_b)
            frac_r = safe_pearson(frac_a, frac_b)
            mean_d = float(np.linalg.norm(mean_a - mean_b))
            frac_d = float(np.linalg.norm(frac_a - frac_b))
            sh_mean = shuffled_trace_similarity(mean_a, mean_b, opts.n_shuffles, rng)
            sh_frac = shuffled_trace_similarity(frac_a, frac_b, opts.n_shuffles, rng)
            pair_name = f"{a['name']}__vs__{b['name']}"
            pairs.append(
                {
                    "pair": pair_name,
                    "source": a["name"],
                    "target": b["name"],
                    "mean_trace_similarity": mean_r,
                    "fraction_active_similarity": frac_r,
                    "mean_trace_euclidean": mean_d,
                    "fraction_active_euclidean": frac_d,
                    "mean_trace_shuffle_mean": np.nanmean(sh_mean),
                    "mean_trace_shuffle_std": np.nanstd(sh_mean, ddof=1),
                    "fraction_active_shuffle_mean": np.nanmean(sh_frac),
                    "fraction_active_shuffle_std": np.nanstd(sh_frac, ddof=1),
                    "mean_trace_effect_over_shuffle": effect_over_shuffle(mean_r, sh_mean),
                    "fraction_active_effect_over_shuffle": effect_over_shuffle(frac_r, sh_frac),
                    "mean_trace_shuffle_p": empirical_upper_p(mean_r, sh_mean),
                    "fraction_active_shuffle_p": empirical_upper_p(frac_r, sh_frac),
                }
            )
            mean_sim[i, j] = mean_sim[j, i] = mean_r
            frac_sim[i, j] = frac_sim[j, i] = frac_r
            mean_dist[i, j] = mean_dist[j, i] = mean_d
            frac_dist[i, j] = frac_dist[j, i] = frac_d
            mean_shuffle.append(sh_mean)
            frac_shuffle.append(sh_frac)

    return {
        "names": [r["name"] for r in records],
        "pairs": pairs,
        "mean_trace_similarity_matrix": mean_sim,
        "fraction_active_similarity_matrix": frac_sim,
        "mean_trace_euclidean_matrix": mean_dist,
        "fraction_active_euclidean_matrix": frac_dist,
        "mean_trace_shuffle_similarity": np.asarray(mean_shuffle),
        "fraction_active_shuffle_similarity": np.asarray(frac_shuffle),
    }


def shuffled_trace_similarity(a: np.ndarray, b: np.ndarray, n_shuffles: int, rng: np.random.Generator) -> np.ndarray:
    out = np.full(n_shuffles, np.nan)
    for idx in range(n_shuffles):
        out[idx] = safe_pearson(rng.permutation(a), rng.permutation(b))
    return out


def compare_distributional_modulation(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = []
    n = len(records)
    hist_corr = np.full((n, n), np.nan)
    hist_js = np.full((n, n), np.nan)
    amp_wasserstein = np.full((n, n), np.nan)
    amp_ks_p = np.full((n, n), np.nan)
    np.fill_diagonal(hist_corr, 1.0)
    np.fill_diagonal(hist_js, 0.0)
    np.fill_diagonal(amp_wasserstein, 0.0)

    for i, a in enumerate(records):
        for j, b in enumerate(records):
            if j <= i:
                continue
            hc = safe_pearson(a["preferred_hist"], b["preferred_hist"])
            js = float(jensenshannon(a["preferred_hist"], b["preferred_hist"]))
            wd = float(wasserstein_distance(a["modulation_amplitude"], b["modulation_amplitude"]))
            ks = ks_2samp(a["modulation_amplitude"], b["modulation_amplitude"])
            pair = {
                "pair": f"{a['name']}__vs__{b['name']}",
                "source": a["name"],
                "target": b["name"],
                "preferred_hist_correlation": hc,
                "preferred_hist_js_distance": js,
                "modulation_wasserstein_distance": wd,
                "modulation_ks_stat": float(ks.statistic),
                "modulation_ks_p": float(ks.pvalue),
            }
            pairs.append(pair)
            hist_corr[i, j] = hist_corr[j, i] = hc
            hist_js[i, j] = hist_js[j, i] = js
            amp_wasserstein[i, j] = amp_wasserstein[j, i] = wd
            amp_ks_p[i, j] = amp_ks_p[j, i] = ks.pvalue
    return {
        "names": [r["name"] for r in records],
        "pairs": pairs,
        "preferred_hist_correlation_matrix": hist_corr,
        "preferred_hist_js_matrix": hist_js,
        "modulation_wasserstein_matrix": amp_wasserstein,
        "modulation_ks_p_matrix": amp_ks_p,
    }


def compare_cebra_geometry(
    records: Sequence[Mapping[str, Any]], opts: PopulationGeometryOptions, rng: np.random.Generator
) -> Dict[str, Any]:
    pairs = []
    n = len(records)
    sim = np.full((n, n), np.nan)
    proc = np.full((n, n), np.nan)
    np.fill_diagonal(sim, 1.0)
    np.fill_diagonal(proc, 0.0)
    shuffle = []
    for i, a in enumerate(records):
        for j, b in enumerate(records):
            if j <= i:
                continue
            r = safe_pearson(a["rdm_vector"], b["rdm_vector"])
            sh = shuffled_cebra_geometry_similarity(a["rdm"], b["rdm"], opts.n_shuffles, rng)
            disparity = procrustes_disparity(a["trajectory"], b["trajectory"]) if opts.procrustes_cebra else np.nan
            pairs.append(
                {
                    "pair": f"{a['name']}__vs__{b['name']}",
                    "source": a["name"],
                    "target": b["name"],
                    "cebra_geometry_similarity": r,
                    "cebra_geometry_shuffle_mean": np.nanmean(sh),
                    "cebra_geometry_shuffle_std": np.nanstd(sh, ddof=1),
                    "cebra_geometry_effect_over_shuffle": effect_over_shuffle(r, sh),
                    "cebra_geometry_shuffle_p": empirical_upper_p(r, sh),
                    "cebra_procrustes_disparity": disparity,
                }
            )
            sim[i, j] = sim[j, i] = r
            proc[i, j] = proc[j, i] = disparity
            shuffle.append(sh)
    return {
        "names": [r["name"] for r in records],
        "pairs": pairs,
        "cebra_geometry_similarity_matrix": sim,
        "cebra_geometry_shuffle_similarity": np.asarray(shuffle),
        "cebra_procrustes_disparity_matrix": proc,
    }


def shuffled_cebra_geometry_similarity(
    rdm_a: np.ndarray, rdm_b: np.ndarray, n_shuffles: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.full(n_shuffles, np.nan)
    for idx in range(n_shuffles):
        pa = rng.permutation(rdm_a.shape[0])
        pb = rng.permutation(rdm_b.shape[0])
        out[idx] = safe_pearson(upper_triangle(rdm_a[pa][:, pa]), upper_triangle(rdm_b[pb][:, pb]))
    return out


def procrustes_disparity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    d = min(a.shape[1], b.shape[1])
    a = a[:n, :d]
    b = b[:n, :d]
    a0 = a - np.mean(a, axis=0, keepdims=True)
    b0 = b - np.mean(b, axis=0, keepdims=True)
    norm = np.linalg.norm(a0)
    if not np.isfinite(norm) or norm == 0:
        norm = 1.0
    a0 = a0 / norm
    b0 = b0 / norm
    rotation, _ = orthogonal_procrustes(a0, b0)
    return float(np.mean(np.sum((a0 @ rotation - b0) ** 2, axis=1)))


def compare_raw_trace_to_cebra_geometry(
    raw_trace: Mapping[str, Any], cebra_geometry: Mapping[str, Any]
) -> Dict[str, Any]:
    raw_pairs = {p["pair"]: p for p in raw_trace["pairs"]}
    cebra_pairs = {p["pair"]: p for p in cebra_geometry["pairs"]}
    shared = sorted(set(raw_pairs) & set(cebra_pairs))
    stats = {"shared_pairs": shared}
    comparisons = {
        "mean_trace_similarity_vs_cebra_geometry": ("mean_trace_similarity", "cebra_geometry_similarity"),
        "fraction_active_similarity_vs_cebra_geometry": ("fraction_active_similarity", "cebra_geometry_similarity"),
        "mean_trace_effect_vs_cebra_effect": ("mean_trace_effect_over_shuffle", "cebra_geometry_effect_over_shuffle"),
        "fraction_active_effect_vs_cebra_effect": ("fraction_active_effect_over_shuffle", "cebra_geometry_effect_over_shuffle"),
    }
    for name, (raw_metric, cebra_metric) in comparisons.items():
        raw = np.asarray([raw_pairs[p][raw_metric] for p in shared], dtype=float)
        cebra = np.asarray([cebra_pairs[p][cebra_metric] for p in shared], dtype=float)
        stats[name] = paired_stats(raw, cebra)
    return stats


def paired_stats(raw: np.ndarray, cebra: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(raw) & np.isfinite(cebra)
    raw = raw[finite]
    cebra = cebra[finite]
    if len(raw) > 1:
        t = ttest_rel(raw, cebra)
        try:
            w = wilcoxon(cebra - raw)
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        t_stat, t_p = float(t.statistic), float(t.pvalue)
    else:
        t_stat = t_p = w_stat = w_p = np.nan
    return {
        "raw_mean": float(np.nanmean(raw)) if len(raw) else np.nan,
        "raw_sem": sem(raw),
        "cebra_mean": float(np.nanmean(cebra)) if len(cebra) else np.nan,
        "cebra_sem": sem(cebra),
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "n_pairs": int(len(raw)),
    }


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 2 or np.nanstd(a[valid]) == 0 or np.nanstd(b[valid]) == 0:
        return np.nan
    return float(pearsonr(a[valid], b[valid]).statistic)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def effect_over_shuffle(real: float, shuffle_values: np.ndarray) -> float:
    shuffle_values = np.asarray(shuffle_values, dtype=float)
    shuffle_values = shuffle_values[np.isfinite(shuffle_values)]
    if len(shuffle_values) < 2:
        return np.nan
    sd = np.nanstd(shuffle_values, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float((real - np.nanmean(shuffle_values)) / sd)


def empirical_upper_p(real: float, shuffle_values: np.ndarray) -> float:
    shuffle_values = np.asarray(shuffle_values, dtype=float)
    shuffle_values = shuffle_values[np.isfinite(shuffle_values)]
    if not np.isfinite(real) or len(shuffle_values) == 0:
        return np.nan
    return float((np.sum(shuffle_values >= real) + 1) / (len(shuffle_values) + 1))


def sem(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def save_outputs(results: Mapping[str, Any], opts: PopulationGeometryOptions) -> None:
    write_records_npz(results, opts)
    write_pair_csv(opts.output_dir / "raw_trace_pairwise_similarity.csv", results["raw_trace"]["pairs"])
    write_pair_csv(opts.output_dir / "distributional_similarity_metrics.csv", results["distributional"]["pairs"])
    write_pair_csv(opts.output_dir / "cebra_geometry_pairwise_similarity.csv", results["cebra_geometry"]["pairs"])
    write_stats_csv(opts.output_dir / "population_trace_vs_cebra_geometry_summary.csv", results["paired_stats"])
    write_matrix_csv(
        opts.output_dir / "raw_mean_trace_similarity_matrix.csv",
        results["raw_trace"]["names"],
        results["raw_trace"]["mean_trace_similarity_matrix"],
    )
    write_matrix_csv(
        opts.output_dir / "raw_fraction_active_similarity_matrix.csv",
        results["raw_trace"]["names"],
        results["raw_trace"]["fraction_active_similarity_matrix"],
    )
    write_matrix_csv(
        opts.output_dir / "cebra_geometry_similarity_matrix.csv",
        results["cebra_geometry"]["names"],
        results["cebra_geometry"]["cebra_geometry_similarity_matrix"],
    )


def write_records_npz(results: Mapping[str, Any], opts: PopulationGeometryOptions) -> None:
    raw_records = results["raw_records"]
    cebra_records = results["cebra_records"]
    np.savez_compressed(
        opts.output_dir / "population_trace_vs_cebra_geometry_results.npz",
        animal_names=np.asarray([r["name"] for r in raw_records], dtype=object),
        raw_mean_trace_by_animal=np.asarray([r["raw_mean_trace_bins"] for r in raw_records], dtype=float),
        raw_mean_trace_z_by_animal=np.asarray([r["raw_mean_trace_bins_z"] for r in raw_records], dtype=float),
        raw_fraction_active_trace_by_animal=np.asarray([r["fraction_active_trace_bins"] for r in raw_records], dtype=float),
        raw_fraction_active_trace_z_by_animal=np.asarray([r["fraction_active_trace_bins_z"] for r in raw_records], dtype=float),
        preferred_bin_by_neuron=np.asarray([r["preferred_bins"] for r in raw_records], dtype=object),
        preferred_bin_hist_by_animal=np.asarray([r["preferred_hist"] for r in raw_records], dtype=float),
        modulation_amplitude_by_neuron=np.asarray([r["modulation_amplitude"] for r in raw_records], dtype=object),
        cebra_task_trajectory_by_animal=np.asarray([r["trajectory"] for r in cebra_records], dtype=object),
        cebra_rdm_by_animal=np.asarray([r["rdm"] for r in cebra_records], dtype=object),
        raw_mean_trace_shuffle_similarity=results["raw_trace"]["mean_trace_shuffle_similarity"],
        raw_fraction_active_shuffle_similarity=results["raw_trace"]["fraction_active_shuffle_similarity"],
        cebra_geometry_shuffle_similarity=results["cebra_geometry"]["cebra_geometry_shuffle_similarity"],
    )


def write_pair_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(path: Path, stats: Mapping[str, Any]) -> None:
    rows = []
    for metric, values in stats.items():
        if metric == "shared_pairs":
            continue
        row = {"metric": metric}
        row.update(values)
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, names: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])


def plot_population_trace_vs_cebra_geometry(results: Mapping[str, Any], opts: PopulationGeometryOptions) -> None:
    sns.set_theme(style="white", context="paper", font="Arial")
    raw_records = results["raw_records"]
    cebra_records = results["cebra_records"]
    raw_pairs = results["raw_trace"]["pairs"]
    cebra_pairs = results["cebra_geometry"]["pairs"]
    paired_stats = results["paired_stats"]

    fig = plt.figure(figsize=(12.0, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_a = fig.add_subplot(gs[0, 0])
    for record in raw_records:
        ax_a.plot(np.arange(1, len(record["raw_mean_trace_bins_z"]) + 1), record["raw_mean_trace_bins_z"], "-o", linewidth=1.2, markersize=4, label=record["rat_name"])
    ax_a.set_title("A  Gross population calcium dynamics", loc="left")
    ax_a.set_xlabel("CSUS5 task bin")
    ax_a.set_ylabel("Amplitude-normalized calcium")
    ax_a.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    for record in raw_records:
        ax_b.plot(np.arange(1, len(record["fraction_active_trace_bins"]) + 1), record["fraction_active_trace_bins"], "-o", linewidth=1.2, markersize=4, label=record["rat_name"])
    ax_b.set_title("B  Population recruitment over task time", loc="left")
    ax_b.set_xlabel("CSUS5 task bin")
    ax_b.set_ylabel("Fraction active")
    sns.despine(ax=ax_b)

    ax_c = fig.add_subplot(gs[0, 2])
    hist = np.vstack([r["preferred_hist"] for r in raw_records])
    bottom = np.zeros(hist.shape[1])
    x = np.arange(1, hist.shape[1] + 1)
    for idx, record in enumerate(raw_records):
        ax_c.plot(x, record["preferred_hist"], "-o", linewidth=1.1, markersize=4, label=record["rat_name"])
    ax_c.set_title("C  Distribution of task-modulated neurons", loc="left")
    ax_c.set_xlabel("Preferred CSUS5 bin")
    ax_c.set_ylabel("Fraction of neurons")
    sns.despine(ax=ax_c)

    ax_d = fig.add_subplot(gs[1, 0], projection="3d")
    plot_aligned_cebra_trajectories(ax_d, cebra_records)

    ax_e = fig.add_subplot(gs[1, 1])
    plot_similarity_panel(ax_e, raw_pairs, cebra_pairs)

    ax_f = fig.add_subplot(gs[1, 2])
    plot_effect_size_panel(ax_f, raw_pairs, cebra_pairs)

    fig.suptitle("Population calcium summaries vs CEBRA latent task geometry", fontsize=13)
    fig.savefig(opts.output_dir / "population_trace_vs_cebra_geometry_figure.png", dpi=300)
    fig.savefig(opts.output_dir / "population_trace_vs_cebra_geometry_figure.svg")
    plt.close(fig)


def plot_aligned_cebra_trajectories(ax: Any, records: Sequence[Mapping[str, Any]]) -> None:
    """Align all CEBRA trajectories to the first rat for visualization."""
    reference = records[0]["trajectory"]
    ref0 = center_scale(reference)
    colors = sns.color_palette("tab10", n_colors=len(records))
    for idx, record in enumerate(records):
        traj = record["trajectory"]
        traj0 = center_scale(traj)
        if idx > 0:
            d = min(ref0.shape[1], traj0.shape[1])
            rotation, _ = orthogonal_procrustes(traj0[:, :d], ref0[:, :d])
            traj0 = traj0[:, :d] @ rotation
        plot_traj = pad_to_3d(traj0)
        ax.plot(plot_traj[:, 0], plot_traj[:, 1], plot_traj[:, 2], "-o", color=colors[idx], linewidth=1.2, markersize=4, label=record["rat_name"])
    ax.set_title("D  CEBRA latent task trajectories", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(frameon=False, fontsize=7)


def center_scale(values: np.ndarray) -> np.ndarray:
    values = values - np.mean(values, axis=0, keepdims=True)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0:
        norm = 1.0
    return values / norm


def pad_to_3d(values: np.ndarray) -> np.ndarray:
    if values.shape[1] >= 3:
        return values[:, :3]
    return np.hstack([values, np.zeros((values.shape[0], 3 - values.shape[1]))])


def plot_similarity_panel(ax: Any, raw_pairs: Sequence[Mapping[str, Any]], cebra_pairs: Sequence[Mapping[str, Any]]) -> None:
    metrics = [
        ("mean_trace_similarity", "Mean trace"),
        ("fraction_active_similarity", "Fraction active"),
        ("cebra_geometry_similarity", "CEBRA geometry"),
    ]
    pair_maps = {
        "mean_trace_similarity": {p["pair"]: p for p in raw_pairs},
        "fraction_active_similarity": {p["pair"]: p for p in raw_pairs},
        "cebra_geometry_similarity": {p["pair"]: p for p in cebra_pairs},
    }
    all_pairs = sorted(set(pair_maps["mean_trace_similarity"]) & set(pair_maps["cebra_geometry_similarity"]))
    values = []
    for x_idx, (metric, label) in enumerate(metrics):
        vals = np.asarray([pair_maps[metric][p][metric] for p in all_pairs], dtype=float)
        values.append(vals)
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) else []
        ax.scatter(np.full(len(vals), x_idx) + jitter, vals, s=24, color=["#4C78A8", "#72B7B2", "#E45756"][x_idx], edgecolor="white", linewidth=0.4)
        ax.errorbar(x_idx, np.nanmean(vals), yerr=sem(vals), fmt="ks", capsize=4, markersize=5)
    for idx in range(len(all_pairs)):
        ax.plot([0, 1, 2], [values[0][idx], values[1][idx], values[2][idx]], color="0.78", linewidth=0.7, zorder=0)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], rotation=25, ha="right")
    ax.set_ylabel("Pairwise cross-animal similarity")
    ax.set_title("E  Cross-animal similarity", loc="left")
    sns.despine(ax=ax)


def plot_effect_size_panel(ax: Any, raw_pairs: Sequence[Mapping[str, Any]], cebra_pairs: Sequence[Mapping[str, Any]]) -> None:
    metrics = [
        ("mean_trace_effect_over_shuffle", "Mean trace"),
        ("fraction_active_effect_over_shuffle", "Fraction active"),
        ("cebra_geometry_effect_over_shuffle", "CEBRA geometry"),
    ]
    pair_maps = {
        "mean_trace_effect_over_shuffle": {p["pair"]: p for p in raw_pairs},
        "fraction_active_effect_over_shuffle": {p["pair"]: p for p in raw_pairs},
        "cebra_geometry_effect_over_shuffle": {p["pair"]: p for p in cebra_pairs},
    }
    all_pairs = sorted(set(pair_maps["mean_trace_effect_over_shuffle"]) & set(pair_maps["cebra_geometry_effect_over_shuffle"]))
    for x_idx, (metric, label) in enumerate(metrics):
        vals = np.asarray([pair_maps[metric][p][metric] for p in all_pairs], dtype=float)
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) else []
        ax.scatter(np.full(len(vals), x_idx) + jitter, vals, s=24, color=["#4C78A8", "#72B7B2", "#E45756"][x_idx], edgecolor="white", linewidth=0.4)
        ax.errorbar(x_idx, np.nanmean(vals), yerr=sem(vals), fmt="ks", capsize=4, markersize=5)
    ax.axhline(0, color="0.45", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], rotation=25, ha="right")
    ax.set_ylabel("(real - shuffle mean) / shuffle SD")
    ax.set_title("F  Effect size over shuffle", loc="left")
    sns.despine(ax=ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("population_trace_vs_cebra_geometry_output"))
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--activity-threshold-z", type=float, default=1.0)
    parser.add_argument("--cebra-embedding-key", default=None)
    parser.add_argument("--cebra-label-key", default=None)
    parser.add_argument("--cebra-bin-vectors-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = PopulationGeometryOptions(
        n_shuffles=args.n_shuffles,
        random_seed=args.random_seed,
        min_trials=args.min_trials,
        min_neurons=args.min_neurons,
        activity_threshold_z=args.activity_threshold_z,
        output_dir=args.output_dir,
        cebra_embedding_key=args.cebra_embedding_key,
        cebra_label_key=args.cebra_label_key,
        cebra_bin_vectors_key=args.cebra_bin_vectors_key,
    )
    sessions = load_sessions_csv(args.session_csv)
    run_population_trace_vs_cebra_geometry(sessions, opts)


if __name__ == "__main__":
    main()
