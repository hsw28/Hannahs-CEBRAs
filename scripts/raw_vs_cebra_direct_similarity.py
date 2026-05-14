#!/usr/bin/env python3
"""
Direct raw-activity vs CEBRA latent cross-animal similarity.

This script replaces the earlier RDM-vs-RDM framing with direct trajectory
comparisons. Raw calcium is represented as CSUS5 task-bin activity patterns,
reduced with PCA to match the CEBRA latent dimensionality. CEBRA is represented
as task-bin latent trajectories from either sample-level embeddings+labels or
pre-binned geometry-preservation NPZ files such as zB_runs.

Conceptual goal:
  1. Raw neural activity patterns are heterogeneous across animals.
  2. CEBRA latent trajectories reveal a shared task organization.
  3. Cross-animal similarity is stronger in latent space than raw activity
     space when comparing direct trajectories rather than RDMs.
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
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from raw_vs_cebra_geometry_comparison import (
    GeometryOptions,
    SessionSpec,
    compute_raw_task_bin_population_vectors,
    extract_task_aligned_calcium,
    get_animal_data,
    get_calcium_session,
    load_cebra_npz,
    load_sessions_csv,
    make_task_bin_labels,
)


@dataclass
class DirectSimilarityOptions:
    task_scheme: str = "CSUS5"
    win: Tuple[float, float] = (0.0, 2.0)
    pre_cs_win: Tuple[float, float] = (-1.0, 0.0)
    csus5_edges: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    n_components: int = 3
    n_shuffles: int = 500
    random_seed: int = 1
    min_trials: int = 5
    min_neurons: int = 10
    zscore_mode: str = "samples"
    output_dir: Path = Path("raw_vs_cebra_direct_similarity_output")
    cebra_embedding_key: Optional[str] = None
    cebra_label_key: Optional[str] = None
    cebra_bin_vectors_key: Optional[str] = None
    example_pair: Optional[Tuple[str, str]] = None


def run_direct_similarity_analysis(
    sessions: Sequence[SessionSpec],
    opts: Optional[DirectSimilarityOptions] = None,
) -> Dict[str, Any]:
    """Run direct raw PCA vs CEBRA latent cross-animal analysis."""
    opts = opts or DirectSimilarityOptions()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.random_seed)

    print("\nDirect raw-activity vs CEBRA latent similarity")
    print(f"  Sessions listed: {len(sessions)}")
    print(f"  Output dir: {opts.output_dir}")

    raw_records = []
    cebra_records = []
    for idx, session in enumerate(sessions, start=1):
        print(f"  [{idx}/{len(sessions)}] {session.rat_name} {session.date_str}")
        try:
            raw_records.append(compute_raw_direct_record(session, opts))
        except Exception as exc:
            warnings.warn(f"{session.key}: raw direct record failed ({exc}); skipping.")
        try:
            cebra_records.append(compute_cebra_direct_record(session, opts))
        except Exception as exc:
            warnings.warn(f"{session.key}: CEBRA direct record failed ({exc}); skipping.")

    shared_names = sorted(set(r["name"] for r in raw_records) & set(r["name"] for r in cebra_records))
    raw_records = [r for r in raw_records if r["name"] in shared_names]
    cebra_records = [r for r in cebra_records if r["name"] in shared_names]
    raw_records.sort(key=lambda r: r["name"])
    cebra_records.sort(key=lambda r: r["name"])

    raw_pairs = compute_pairwise_direct_metrics(raw_records, opts, rng)
    cebra_pairs = compute_pairwise_direct_metrics(cebra_records, opts, rng)
    stats = compute_paired_statistics(raw_pairs, cebra_pairs)

    results = {
        "opts": opts,
        "sessions": sessions,
        "raw_records": raw_records,
        "cebra_records": cebra_records,
        "raw_pairs": raw_pairs,
        "cebra_pairs": cebra_pairs,
        "stats": stats,
    }
    save_direct_outputs(results, opts)
    plot_direct_similarity_results(results, opts)

    print(f"  Raw valid datasets: {len(raw_records)}")
    print(f"  CEBRA valid datasets: {len(cebra_records)}")
    print(f"  Animal pairs: {len(raw_pairs)}")
    print(f"\nDone. Results saved to {opts.output_dir}")
    return results


def as_geometry_options(opts: DirectSimilarityOptions) -> GeometryOptions:
    """Adapt direct-analysis options to the raw extraction helper API."""
    return GeometryOptions(
        task_schemes=(opts.task_scheme,),
        win=opts.win,
        pre_cs_win=opts.pre_cs_win,
        csus5_edges=opts.csus5_edges,
        zscore_mode=opts.zscore_mode,
        min_trials=opts.min_trials,
        min_neurons=opts.min_neurons,
        cebra_embedding_key=opts.cebra_embedding_key,
        cebra_label_key=opts.cebra_label_key,
        cebra_bin_vectors_key=opts.cebra_bin_vectors_key,
    )


def compute_raw_direct_record(session: SessionSpec, opts: DirectSimilarityOptions) -> Dict[str, Any]:
    """Create one animal/session raw PCA trajectory and trial-level samples."""
    helper_opts = as_geometry_options(opts)
    animal = get_animal_data(session, None)
    calcium_all, calcium_ts, trial_cs = get_calcium_session(animal, session)
    if len(trial_cs) < opts.min_trials:
        raise ValueError(f"only {len(trial_cs)} CS trials")

    align_win = (min(opts.win[0], opts.pre_cs_win[0]), max(opts.win[1], opts.pre_cs_win[1]))
    calcium_aligned, aligned_ts = extract_task_aligned_calcium(calcium_all, calcium_ts, trial_cs, align_win)
    bin_vectors, info = compute_raw_task_bin_population_vectors(
        calcium_aligned, aligned_ts, opts.task_scheme, helper_opts
    )
    if bin_vectors.shape[1] < opts.min_neurons:
        raise ValueError(f"only {bin_vectors.shape[1]} valid neurons")

    sample_vectors, sample_labels = compute_trial_bin_samples(calcium_aligned, aligned_ts, info, opts, helper_opts)
    sample_vectors = sample_vectors[:, info["valid_neurons"]]

    n_components = min(opts.n_components, sample_vectors.shape[0] - 1, sample_vectors.shape[1])
    if n_components < 2:
        raise ValueError("not enough raw samples/features for PCA")
    pca = PCA(n_components=n_components, random_state=opts.random_seed)
    raw_samples_pca = pca.fit_transform(sample_vectors)
    raw_trajectory_pca = pca.transform(bin_vectors)

    return {
        "name": session.key,
        "rat_name": session.rat_name,
        "date_str": session.date_str,
        "trajectory": raw_trajectory_pca,
        "samples": raw_samples_pca,
        "labels": sample_labels,
        "bin_labels": np.arange(1, raw_trajectory_pca.shape[0] + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "source": "raw_pca",
    }


def compute_trial_bin_samples(
    calcium_aligned: np.ndarray,
    aligned_ts: np.ndarray,
    info: Mapping[str, Any],
    opts: DirectSimilarityOptions,
    helper_opts: GeometryOptions,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one raw activity vector per trial and CSUS5 bin."""
    bin_labels, _, _ = make_task_bin_labels(aligned_ts, opts.task_scheme, helper_opts)
    task_mask = np.isfinite(bin_labels)
    samples = calcium_aligned.copy()
    if opts.zscore_mode == "samples":
        samples = zscore_neurons_over_task_samples(samples, task_mask)

    rows = []
    labels = []
    n_bins = len(info["bin_edges"]) - 1
    for trial_idx in range(samples.shape[0]):
        for bin_idx in range(n_bins):
            tmask = bin_labels == (bin_idx + 1)
            if np.any(tmask):
                row = np.nanmean(samples[trial_idx, tmask, :], axis=0)
                if np.all(np.isfinite(row)):
                    rows.append(row)
                    labels.append(bin_idx + 1)
    if not rows:
        raise ValueError("no trial-bin raw samples")
    return np.vstack(rows), np.asarray(labels)


def zscore_neurons_over_task_samples(calcium_aligned: np.ndarray, task_time_mask: np.ndarray) -> np.ndarray:
    """Z-score each neuron over trial/time task samples within one animal."""
    out = calcium_aligned.copy()
    for neuron_idx in range(out.shape[2]):
        vals = out[:, task_time_mask, neuron_idx].reshape(-1)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        out[:, :, neuron_idx] = (out[:, :, neuron_idx] - mu) / sigma
    return out


def compute_cebra_direct_record(session: SessionSpec, opts: DirectSimilarityOptions) -> Dict[str, Any]:
    """Create one CEBRA latent trajectory and sample matrix from an NPZ file."""
    if session.npz_path is None:
        raise ValueError("missing npz_path")
    helper_opts = as_geometry_options(opts)
    embedding, labels, emb_key, label_key = load_cebra_npz(session.npz_path, helper_opts)

    pre_binned = emb_key in {"zB_runs", "zA_runs", "bin_vectors", "binVectors"} or (
        embedding.shape[0] == len(labels) and len(labels) <= 20
    )
    if pre_binned:
        trajectory = embedding[:, : opts.n_components]
        samples, sample_labels = samples_from_prebinned_cebra_npz(session.npz_path, emb_key, opts)
    else:
        keep = labels != 0
        embedding = embedding[keep, : opts.n_components]
        labels = labels[keep]
        bin_labels = np.asarray(sorted(np.unique(labels)))
        trajectory = np.vstack([np.nanmean(embedding[labels == label], axis=0) for label in bin_labels])
        samples, sample_labels = embedding, labels

    if trajectory.shape[1] != opts.n_components:
        raise ValueError(f"CEBRA trajectory has {trajectory.shape[1]} dims, expected {opts.n_components}")

    return {
        "name": session.key,
        "rat_name": session.rat_name,
        "date_str": session.date_str,
        "trajectory": trajectory,
        "samples": samples[:, : opts.n_components],
        "labels": sample_labels,
        "bin_labels": np.arange(1, trajectory.shape[0] + 1),
        "embedding_key": emb_key,
        "label_key": label_key,
        "source": "cebra",
    }


def samples_from_prebinned_cebra_npz(
    npz_path: Path, bin_vectors_key: str, opts: DirectSimilarityOptions
) -> Tuple[np.ndarray, np.ndarray]:
    """Use independent CEBRA runs as repeated samples for each task bin."""
    npz = np.load(npz_path, allow_pickle=True)
    z = np.asarray(npz[bin_vectors_key], dtype=float)
    if z.ndim == 2:
        labels = np.arange(1, z.shape[0] + 1)
        return z[:, : opts.n_components], labels
    if z.ndim != 3:
        raise ValueError(f"{bin_vectors_key} must be 2D or 3D, got {z.shape}")
    n_runs, n_bins, n_dim = z.shape
    samples = z[:, :, : opts.n_components].reshape(n_runs * n_bins, min(opts.n_components, n_dim))
    labels = np.tile(np.arange(1, n_bins + 1), n_runs)
    return samples, labels


def compute_pairwise_direct_metrics(
    records: Sequence[Mapping[str, Any]],
    opts: DirectSimilarityOptions,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Compute direct Procrustes and decoder metrics for all animal pairs."""
    pairs = []
    for i, source in enumerate(records):
        for j, target in enumerate(records):
            if j <= i:
                continue
            real = compare_two_animals_direct(source, target)
            shuffle = shuffle_pair_metrics(source, target, opts, rng)
            pairs.append(
                {
                    "pair": f"{source['name']}__vs__{target['name']}",
                    "source": source["name"],
                    "target": target["name"],
                    **real,
                    "shuffle_disparity_mean": np.nanmean(shuffle["disparity"]),
                    "shuffle_disparity_std": np.nanstd(shuffle["disparity"], ddof=1),
                    "shuffle_trajectory_correlation_mean": np.nanmean(shuffle["trajectory_correlation"]),
                    "shuffle_decoding_accuracy_mean": np.nanmean(shuffle["decoding_accuracy"]),
                    "shuffle_decoding_accuracy_std": np.nanstd(shuffle["decoding_accuracy"], ddof=1),
                    "disparity_effect_over_shuffle": effect_over_shuffle(
                        real["procrustes_disparity"], shuffle["disparity"], lower_is_better=True
                    ),
                    "decoding_effect_over_shuffle": effect_over_shuffle(
                        real["cross_animal_decoding_accuracy"], shuffle["decoding_accuracy"], lower_is_better=False
                    ),
                }
            )
    return pairs


def compare_two_animals_direct(source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, float]:
    """Align source trajectory to target trajectory and decode target bins."""
    source_traj, target_traj = matched_trajectories(source, target)
    alignment = procrustes_align(source_traj, target_traj)
    source_samples_aligned = apply_alignment(source["samples"], alignment)
    target_samples_centered = center_and_scale_samples(target["samples"], alignment["target_center"], alignment["scale"])

    decoder = make_pipeline(StandardScaler(), NearestCentroid())
    decoder.fit(source_samples_aligned, source["labels"])
    predicted = decoder.predict(target_samples_centered)
    decode_acc = accuracy_score(target["labels"], predicted)

    return {
        "procrustes_disparity": alignment["disparity"],
        "trajectory_correlation": alignment["trajectory_correlation"],
        "variance_explained": alignment["variance_explained"],
        "cross_animal_decoding_accuracy": decode_acc,
    }


def matched_trajectories(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return task-bin trajectories restricted to shared labels."""
    source_labels = np.asarray(source["bin_labels"])
    target_labels = np.asarray(target["bin_labels"])
    shared = [label for label in source_labels if label in set(target_labels)]
    source_idx = [np.where(source_labels == label)[0][0] for label in shared]
    target_idx = [np.where(target_labels == label)[0][0] for label in shared]
    return source["trajectory"][source_idx], target["trajectory"][target_idx]


def procrustes_align(source_traj: np.ndarray, target_traj: np.ndarray) -> Dict[str, Any]:
    """Orthogonally align source task trajectory to target task trajectory."""
    if source_traj.shape != target_traj.shape:
        raise ValueError(f"trajectory shapes differ: {source_traj.shape} vs {target_traj.shape}")
    source_center = np.mean(source_traj, axis=0, keepdims=True)
    target_center = np.mean(target_traj, axis=0, keepdims=True)
    source_centered = source_traj - source_center
    target_centered = target_traj - target_center
    scale = np.linalg.norm(source_centered)
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    source_scaled = source_centered / scale
    target_scaled = target_centered / scale

    rotation, _ = orthogonal_procrustes(source_scaled, target_scaled)
    aligned = source_scaled @ rotation
    residual = aligned - target_scaled
    sse = float(np.sum(residual**2))
    sst = float(np.sum((target_scaled - np.mean(target_scaled, axis=0, keepdims=True)) ** 2))
    disparity = float(np.mean(np.sum(residual**2, axis=1)))
    trajectory_corr = safe_pearson(aligned.reshape(-1), target_scaled.reshape(-1))
    variance_explained = float(1.0 - sse / sst) if sst > 0 else np.nan
    return {
        "rotation": rotation,
        "source_center": source_center,
        "target_center": target_center,
        "scale": scale,
        "aligned_source": aligned,
        "target_scaled": target_scaled,
        "disparity": disparity,
        "trajectory_correlation": trajectory_corr,
        "variance_explained": variance_explained,
    }


def apply_alignment(samples: np.ndarray, alignment: Mapping[str, Any]) -> np.ndarray:
    """Apply trajectory-derived centering/scaling/rotation to source samples."""
    return ((samples - alignment["source_center"]) / alignment["scale"]) @ alignment["rotation"]


def center_and_scale_samples(samples: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Place target samples in the same centered/scaled coordinate system."""
    return (samples - center) / scale


def shuffle_pair_metrics(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    opts: DirectSimilarityOptions,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Shuffle target task-bin identity while preserving within-bin samples."""
    out = {
        "disparity": np.full(opts.n_shuffles, np.nan),
        "trajectory_correlation": np.full(opts.n_shuffles, np.nan),
        "decoding_accuracy": np.full(opts.n_shuffles, np.nan),
    }
    for shuffle_idx in range(opts.n_shuffles):
        shuffled_target = shuffle_record_labels_and_trajectory(target, rng)
        metrics = compare_two_animals_direct(source, shuffled_target)
        out["disparity"][shuffle_idx] = metrics["procrustes_disparity"]
        out["trajectory_correlation"][shuffle_idx] = metrics["trajectory_correlation"]
        out["decoding_accuracy"][shuffle_idx] = metrics["cross_animal_decoding_accuracy"]
    return out


def shuffle_record_labels_and_trajectory(record: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Permute task-bin labels as a control while preserving temporal blocks."""
    out = dict(record)
    bin_labels = np.asarray(record["bin_labels"])
    permuted = rng.permutation(bin_labels)
    label_map = dict(zip(bin_labels, permuted))
    out["bin_labels"] = permuted
    out["labels"] = np.asarray([label_map.get(label, label) for label in record["labels"]])
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(pearsonr(x[valid], y[valid]).statistic)


def effect_over_shuffle(real: float, shuffle_values: np.ndarray, lower_is_better: bool) -> float:
    shuffle_values = np.asarray(shuffle_values, dtype=float)
    shuffle_values = shuffle_values[np.isfinite(shuffle_values)]
    if len(shuffle_values) < 2:
        return np.nan
    sd = np.nanstd(shuffle_values, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    diff = np.nanmean(shuffle_values) - real if lower_is_better else real - np.nanmean(shuffle_values)
    return float(diff / sd)


def compute_paired_statistics(raw_pairs: Sequence[Mapping[str, Any]], cebra_pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Paired tests across identical animal pairs."""
    raw_by_pair = {p["pair"]: p for p in raw_pairs}
    cebra_by_pair = {p["pair"]: p for p in cebra_pairs}
    shared = sorted(set(raw_by_pair) & set(cebra_by_pair))
    stats = {"shared_pairs": shared}
    for metric in [
        "procrustes_disparity",
        "trajectory_correlation",
        "variance_explained",
        "cross_animal_decoding_accuracy",
        "disparity_effect_over_shuffle",
        "decoding_effect_over_shuffle",
    ]:
        raw = np.asarray([raw_by_pair[p][metric] for p in shared], dtype=float)
        cebra = np.asarray([cebra_by_pair[p][metric] for p in shared], dtype=float)
        finite = np.isfinite(raw) & np.isfinite(cebra)
        raw = raw[finite]
        cebra = cebra[finite]
        if len(raw) > 1:
            t_p = float(ttest_rel(raw, cebra).pvalue)
            try:
                w_p = float(wilcoxon(cebra - raw).pvalue)
            except ValueError:
                w_p = np.nan
        else:
            t_p = np.nan
            w_p = np.nan
        stats[metric] = {
            "raw_mean": float(np.nanmean(raw)) if len(raw) else np.nan,
            "raw_sem": sem(raw),
            "cebra_mean": float(np.nanmean(cebra)) if len(cebra) else np.nan,
            "cebra_sem": sem(cebra),
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
            "n_pairs": int(len(raw)),
        }
    return stats


def sem(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def save_direct_outputs(results: Mapping[str, Any], opts: DirectSimilarityOptions) -> None:
    """Save pair metrics, summary statistics, and compact NPZ artifacts."""
    write_pair_csv(opts.output_dir / "raw_direct_pair_metrics.csv", results["raw_pairs"])
    write_pair_csv(opts.output_dir / "cebra_direct_pair_metrics.csv", results["cebra_pairs"])
    write_stats_csv(opts.output_dir / "raw_vs_cebra_direct_similarity_summary.csv", results["stats"])
    np.savez_compressed(
        opts.output_dir / "raw_vs_cebra_direct_similarity_results.npz",
        raw_pair_metrics=np.asarray(results["raw_pairs"], dtype=object),
        cebra_pair_metrics=np.asarray(results["cebra_pairs"], dtype=object),
        stats=np.asarray(results["stats"], dtype=object),
        raw_names=np.asarray([r["name"] for r in results["raw_records"]], dtype=object),
        cebra_names=np.asarray([r["name"] for r in results["cebra_records"]], dtype=object),
        raw_trajectories=np.asarray([r["trajectory"] for r in results["raw_records"]], dtype=object),
        cebra_trajectories=np.asarray([r["trajectory"] for r in results["cebra_records"]], dtype=object),
        raw_sample_labels=np.asarray([r["labels"] for r in results["raw_records"]], dtype=object),
        cebra_sample_labels=np.asarray([r["labels"] for r in results["cebra_records"]], dtype=object),
    )


def write_pair_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "pair",
        "source",
        "target",
        "procrustes_disparity",
        "trajectory_correlation",
        "variance_explained",
        "cross_animal_decoding_accuracy",
        "shuffle_disparity_mean",
        "shuffle_disparity_std",
        "shuffle_decoding_accuracy_mean",
        "shuffle_decoding_accuracy_std",
        "disparity_effect_over_shuffle",
        "decoding_effect_over_shuffle",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)


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


def plot_direct_similarity_results(results: Mapping[str, Any], opts: DirectSimilarityOptions) -> None:
    """Create publication-style multi-panel direct-similarity figure."""
    sns.set_theme(style="white", context="paper", font="Arial")
    palette = {"Raw PCA": "#4C78A8", "CEBRA": "#E45756", "Shuffle": "0.70"}

    raw_records = results["raw_records"]
    cebra_records = results["cebra_records"]
    raw_pairs = results["raw_pairs"]
    cebra_pairs = results["cebra_pairs"]
    stats = results["stats"]
    example_raw, example_cebra = choose_example_pair(raw_records, cebra_records, raw_pairs, cebra_pairs, opts)

    fig = plt.figure(figsize=(11.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    plot_example_alignment(ax_a, example_raw, "A  Raw PCA trajectories", palette["Raw PCA"])

    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    plot_example_alignment(ax_b, example_cebra, "B  CEBRA latent trajectories", palette["CEBRA"])

    ax_c = fig.add_subplot(gs[0, 2])
    plot_paired_metric(
        ax_c,
        raw_pairs,
        cebra_pairs,
        "procrustes_disparity",
        "C  Procrustes disparity",
        "Disparity (lower better)",
        palette,
    )

    ax_d = fig.add_subplot(gs[1, 0])
    plot_paired_metric(
        ax_d,
        raw_pairs,
        cebra_pairs,
        "cross_animal_decoding_accuracy",
        "D  Cross-animal decoding",
        "Accuracy",
        palette,
    )
    ax_d.axhline(1 / 5, color="0.45", linestyle="--", linewidth=1)

    ax_e = fig.add_subplot(gs[1, 1])
    plot_paired_metric(
        ax_e,
        raw_pairs,
        cebra_pairs,
        "disparity_effect_over_shuffle",
        "E  Alignment effect over shuffle",
        "Effect size",
        palette,
    )
    ax_e.axhline(0, color="0.45", linestyle="--", linewidth=1)

    ax_f = fig.add_subplot(gs[1, 2])
    plot_summary_text(ax_f, stats)

    fig.suptitle("Direct cross-animal similarity: raw activity vs CEBRA latent task structure", fontsize=13)
    fig.savefig(opts.output_dir / "raw_vs_cebra_direct_similarity_figure.png", dpi=300)
    fig.savefig(opts.output_dir / "raw_vs_cebra_direct_similarity_figure.svg")
    plt.close(fig)


def choose_example_pair(
    raw_records: Sequence[Mapping[str, Any]],
    cebra_records: Sequence[Mapping[str, Any]],
    raw_pairs: Sequence[Mapping[str, Any]],
    cebra_pairs: Sequence[Mapping[str, Any]],
    opts: DirectSimilarityOptions,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick one animal pair for trajectory panels."""
    if opts.example_pair:
        source_name, target_name = opts.example_pair
    else:
        source_name, target_name = raw_pairs[0]["source"], raw_pairs[0]["target"]
    raw_by_name = {r["name"]: r for r in raw_records}
    cebra_by_name = {r["name"]: r for r in cebra_records}
    return (
        make_example_alignment(raw_by_name[source_name], raw_by_name[target_name]),
        make_example_alignment(cebra_by_name[source_name], cebra_by_name[target_name]),
    )


def make_example_alignment(source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    source_traj, target_traj = matched_trajectories(source, target)
    alignment = procrustes_align(source_traj, target_traj)
    return {
        "source_name": source["name"],
        "target_name": target["name"],
        "source": alignment["aligned_source"],
        "target": alignment["target_scaled"],
        "disparity": alignment["disparity"],
        "corr": alignment["trajectory_correlation"],
    }


def plot_example_alignment(ax: Any, example: Mapping[str, Any], title: str, color: str) -> None:
    """Show one aligned task trajectory pair in 3D."""
    source = pad_to_3d(np.asarray(example["source"]))
    target = pad_to_3d(np.asarray(example["target"]))
    bins = np.arange(1, source.shape[0] + 1)
    ax.plot(source[:, 0], source[:, 1], source[:, 2], "-o", color=color, alpha=0.45, label="Source rat")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], "-o", color="black", alpha=0.85, label="Target rat")
    for idx, label in enumerate(bins):
        ax.text(target[idx, 0], target[idx, 1], target[idx, 2], str(label), fontsize=7)
    ax.set_title(f"{title}\nr={example['corr']:.2f}, disp={example['disparity']:.3f}", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def pad_to_3d(values: np.ndarray) -> np.ndarray:
    if values.shape[1] >= 3:
        return values[:, :3]
    pad = np.zeros((values.shape[0], 3 - values.shape[1]))
    return np.hstack([values, pad])


def plot_paired_metric(
    ax: Any,
    raw_pairs: Sequence[Mapping[str, Any]],
    cebra_pairs: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    palette: Mapping[str, str],
) -> None:
    raw_by_pair = {p["pair"]: p for p in raw_pairs}
    cebra_by_pair = {p["pair"]: p for p in cebra_pairs}
    shared = sorted(set(raw_by_pair) & set(cebra_by_pair))
    raw_values = np.asarray([raw_by_pair[p][metric] for p in shared], dtype=float)
    cebra_values = np.asarray([cebra_by_pair[p][metric] for p in shared], dtype=float)
    for rv, cv in zip(raw_values, cebra_values):
        ax.plot([0, 1], [rv, cv], color="0.72", linewidth=0.9, zorder=1)
    ax.scatter(np.zeros_like(raw_values), raw_values, s=28, color=palette["Raw PCA"], edgecolor="white", linewidth=0.5, zorder=2)
    ax.scatter(np.ones_like(cebra_values), cebra_values, s=28, color=palette["CEBRA"], edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar([0, 1], [np.nanmean(raw_values), np.nanmean(cebra_values)], yerr=[sem(raw_values), sem(cebra_values)], fmt="ks", capsize=4, markersize=5, zorder=3)
    ax.set_xticks([0, 1], ["Raw PCA", "CEBRA"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    sns.despine(ax=ax)


def plot_summary_text(ax: Any, stats: Mapping[str, Any]) -> None:
    ax.axis("off")
    lines = ["Statistics across paired animal pairs"]
    labels = [
        ("procrustes_disparity", "Disparity"),
        ("trajectory_correlation", "Trajectory r"),
        ("variance_explained", "Variance explained"),
        ("cross_animal_decoding_accuracy", "Decoding"),
    ]
    for key, label in labels:
        s = stats[key]
        lines.append(
            f"{label}: raw {s['raw_mean']:.3f} +/- {s['raw_sem']:.3f}, "
            f"CEBRA {s['cebra_mean']:.3f} +/- {s['cebra_sem']:.3f}, "
            f"p={s['paired_t_p']:.3g}"
        )
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("raw_vs_cebra_direct_similarity_output"))
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--zscore-mode", choices=["samples", "binMeans", "none"], default="samples")
    parser.add_argument("--cebra-embedding-key", default=None)
    parser.add_argument("--cebra-label-key", default=None)
    parser.add_argument("--cebra-bin-vectors-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = DirectSimilarityOptions(
        n_components=args.n_components,
        n_shuffles=args.n_shuffles,
        random_seed=args.random_seed,
        min_trials=args.min_trials,
        min_neurons=args.min_neurons,
        zscore_mode=args.zscore_mode,
        output_dir=args.output_dir,
        cebra_embedding_key=args.cebra_embedding_key,
        cebra_label_key=args.cebra_label_key,
        cebra_bin_vectors_key=args.cebra_bin_vectors_key,
    )
    sessions = load_sessions_csv(args.session_csv)
    run_direct_similarity_analysis(sessions, opts)


if __name__ == "__main__":
    main()
