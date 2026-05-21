#!/usr/bin/env python3
"""
Trial-bin population-summary calcium features vs CEBRA latent decoding.

Unit of analysis:
    one sample = one trial x one CSUS5 task bin within one session

Raw calcium is collapsed across neurons into neuron-identity-free features:
    population_mean, fraction_active, population_std, quantiles, total_events_per_neuron

CEBRA is represented as latent samples with CSUS5 labels. For sample-level
embeddings_and_labels.npz files, samples are embedding rows. For pre-binned
geometry NPZ files such as zB_runs, each independent run x task bin is treated
as one labeled latent sample. If an external cross-rat CEBRA decoder summary is
provided, those real decoder results are used instead of re-decoding local NPZs.

The central test is cross-animal decoding on matched ordered session pairs:
    train session A -> test session B, where A and B are different animals.
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
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_rel, wasserstein_distance, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from raw_vs_cebra_geometry_comparison import (
    GeometryOptions,
    SessionSpec,
    extract_task_aligned_calcium,
    get_animal_data,
    get_calcium_session,
    load_sessions_csv,
    make_task_bin_labels,
)


@dataclass
class TrialBinDecodeOptions:
    task_scheme: str = "CSUS5"
    win: Tuple[float, float] = (0.0, 2.0)
    pre_cs_win: Tuple[float, float] = (-1.0, 0.0)
    csus5_edges: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    min_trials: int = 5
    min_neurons: int = 10
    active_threshold: float = 0.0
    active_use_zscore: bool = False
    baseline_subtract_raw: bool = False
    include_quantiles: bool = True
    classifier: str = "logreg"
    n_shuffles: int = 500
    random_seed: int = 1
    output_dir: Path = Path("trial_bin_population_summary_vs_cebra_decoding_output")
    cebra_embedding_key: Optional[str] = None
    cebra_label_key: Optional[str] = None
    cebra_bin_vectors_key: Optional[str] = None
    cebra_decoder_summary_csv: Optional[Path] = None
    cebra_decoder_task_scheme: str = "CSUS5"
    cebra_decoder_dim: int = 3
    cebra_decoder_comparison: Optional[str] = None
    run_within_session_cv: bool = True


def run_trial_bin_population_summary_vs_cebra_decoding(
    sessions: Sequence[SessionSpec],
    opts: Optional[TrialBinDecodeOptions] = None,
) -> Dict[str, Any]:
    """Run trial-bin raw-summary and CEBRA cross-animal decoding."""
    opts = opts or TrialBinDecodeOptions()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.random_seed)

    print("\nTrial-bin population-summary calcium vs CEBRA decoding")
    print(f"  Sessions listed: {len(sessions)}")
    print(f"  Output dir: {opts.output_dir}")

    raw_sessions = []
    cebra_sessions = []
    for idx, session in enumerate(sessions, start=1):
        print(f"  [{idx}/{len(sessions)}] {session.rat_name} {session.date_str}")
        try:
            raw_sessions.append(extract_raw_summary_features(session, opts))
        except Exception as exc:
            warnings.warn(f"{session.key}: raw-summary feature extraction failed ({exc}); skipping.")
        if opts.cebra_decoder_summary_csv is None:
            try:
                cebra_sessions.append(extract_cebra_features(session, opts))
            except Exception as exc:
                warnings.warn(f"{session.key}: CEBRA feature extraction failed ({exc}); skipping.")

    if opts.cebra_decoder_summary_csv is None:
        shared = sorted(set(s["session_key"] for s in raw_sessions) & set(s["session_key"] for s in cebra_sessions))
        raw_sessions = sorted([s for s in raw_sessions if s["session_key"] in shared], key=lambda s: s["session_key"])
        cebra_sessions = sorted([s for s in cebra_sessions if s["session_key"] in shared], key=lambda s: s["session_key"])
    else:
        shared = sorted(s["session_key"] for s in raw_sessions)
        raw_sessions = sorted(raw_sessions, key=lambda s: s["session_key"])

    ordered_pairs = build_ordered_session_pairs(raw_sessions)
    raw_decode = decode_ordered_pairs(raw_sessions, ordered_pairs, opts, rng, feature_family="raw_summary")
    if opts.cebra_decoder_summary_csv is None:
        cebra_decode = decode_ordered_pairs(cebra_sessions, ordered_pairs, opts, rng, feature_family="cebra")
    else:
        cebra_decode = load_real_cebra_decoder_summary(opts.cebra_decoder_summary_csv, ordered_pairs, opts)
    session_similarity = compute_session_trace_and_distribution_similarity(raw_sessions, ordered_pairs, opts, rng)
    within_session = compute_within_session_controls(raw_sessions, cebra_sessions, opts) if opts.run_within_session_cv else []
    stats = paired_decode_statistics(raw_decode, cebra_decode)

    results = {
        "opts": opts,
        "raw_sessions": raw_sessions,
        "cebra_sessions": cebra_sessions,
        "ordered_pairs": ordered_pairs,
        "raw_decode": raw_decode,
        "cebra_decode": cebra_decode,
        "session_similarity": session_similarity,
        "within_session": within_session,
        "stats": stats,
    }
    save_outputs(results, opts)
    plot_trial_bin_decoding_results(results, opts)

    print(f"  Valid matched sessions: {len(shared)}")
    print(f"  Ordered cross-animal session pairs: {len(ordered_pairs)}")
    print(f"\nDone. Results saved to {opts.output_dir}")
    return results


def helper_geometry_opts(opts: TrialBinDecodeOptions) -> GeometryOptions:
    return GeometryOptions(
        task_schemes=(opts.task_scheme,),
        win=opts.win,
        pre_cs_win=opts.pre_cs_win,
        csus5_edges=opts.csus5_edges,
        min_trials=opts.min_trials,
        min_neurons=opts.min_neurons,
    )


def extract_raw_summary_features(session: SessionSpec, opts: TrialBinDecodeOptions) -> Dict[str, Any]:
    """Extract neuron-identity-free features for each trial x task bin."""
    animal = get_animal_data(session, None)
    calcium_all, calcium_ts, trial_cs = get_calcium_session(animal, session)
    if len(trial_cs) < opts.min_trials:
        raise ValueError(f"only {len(trial_cs)} trials")

    align_win = (min(opts.win[0], opts.pre_cs_win[0]), max(opts.win[1], opts.pre_cs_win[1]))
    aligned, aligned_ts = extract_task_aligned_calcium(calcium_all, calcium_ts, trial_cs, align_win)
    if aligned.shape[2] < opts.min_neurons:
        raise ValueError(f"only {aligned.shape[2]} neurons")

    helper_opts = helper_geometry_opts(opts)
    bin_labels_time, bin_edges, bin_names = make_task_bin_labels(aligned_ts, opts.task_scheme, helper_opts)
    if opts.baseline_subtract_raw:
        aligned = subtract_trial_neuron_baseline(aligned, aligned_ts, opts.pre_cs_win)
    z_aligned = zscore_neurons_over_task_samples(aligned, np.isfinite(bin_labels_time))
    activity_source = z_aligned if opts.active_use_zscore else aligned

    rows = []
    labels = []
    trial_ids = []
    trial_trace_mean = []
    trial_trace_fraction = []
    n_bins = len(bin_edges) - 1
    for trial_idx in range(aligned.shape[0]):
        mean_trace = []
        fraction_trace = []
        for bin_idx in range(1, n_bins + 1):
            mask = bin_labels_time == bin_idx
            values = aligned[trial_idx, mask, :]
            activity_values = activity_source[trial_idx, mask, :]
            neuron_bin_values = np.nanmean(values, axis=0)
            activity_neuron_values = np.nanmean(activity_values, axis=0)
            feature_row = raw_summary_feature_row(neuron_bin_values, activity_neuron_values, opts)
            if np.all(np.isfinite(feature_row)):
                rows.append(feature_row)
                labels.append(bin_idx)
                trial_ids.append(trial_idx)
            mean_trace.append(np.nanmean(neuron_bin_values))
            fraction_trace.append(np.nanmean(activity_neuron_values > opts.active_threshold))
        trial_trace_mean.append(mean_trace)
        trial_trace_fraction.append(fraction_trace)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    if X.shape[0] == 0:
        raise ValueError("no valid trial-bin samples")

    return {
        "session_key": session.key,
        "animal_id": session.rat_name,
        "session_id": session.session_name or session.date_str,
        "date_str": session.date_str,
        "X": X,
        "y": y,
        "trial_id": np.asarray(trial_ids, dtype=int),
        "feature_names": raw_summary_feature_names(opts),
        "trial_trace_population_mean": np.asarray(trial_trace_mean, dtype=float),
        "trial_trace_fraction_active": np.asarray(trial_trace_fraction, dtype=float),
        "trial_averaged_population_mean_trace": np.nanmean(np.asarray(trial_trace_mean, dtype=float), axis=0),
        "trial_averaged_fraction_active_trace": np.nanmean(np.asarray(trial_trace_fraction, dtype=float), axis=0),
        "bin_names": bin_names,
    }


def raw_summary_feature_names(opts: TrialBinDecodeOptions) -> List[str]:
    names = ["population_mean", "fraction_active", "population_std"]
    if opts.include_quantiles:
        names.extend(["q25", "q50", "q75", "q90"])
    names.append("total_events_per_neuron")
    return names


def raw_summary_feature_row(values: np.ndarray, activity_values: np.ndarray, opts: TrialBinDecodeOptions) -> np.ndarray:
    """Build one trial x bin population-summary feature vector."""
    active = activity_values > opts.active_threshold
    features = [
        np.nanmean(values),
        np.nanmean(active),
        np.nanstd(values),
    ]
    if opts.include_quantiles:
        features.extend(np.nanpercentile(values, [25, 50, 75, 90]).tolist())
    # This is normalized by neuron count, so it is comparable across animals.
    features.append(np.nansum(np.maximum(values, 0)) / values.size)
    return np.asarray(features, dtype=float)


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


def subtract_trial_neuron_baseline(
    aligned: np.ndarray,
    aligned_ts: np.ndarray,
    baseline_win: Tuple[float, float],
) -> np.ndarray:
    """Subtract each trial x neuron's pre-CS mean before population summaries."""
    baseline_mask = (aligned_ts >= baseline_win[0]) & (aligned_ts < baseline_win[1])
    if np.sum(baseline_mask) == 0:
        warnings.warn(
            f"No baseline samples found in {baseline_win}; raw baseline subtraction was skipped."
        )
        return aligned
    baseline = np.nanmean(aligned[:, baseline_mask, :], axis=1, keepdims=True)
    return aligned - baseline


def extract_cebra_features(session: SessionSpec, opts: TrialBinDecodeOptions) -> Dict[str, Any]:
    """Load CEBRA latent samples and CSUS5 labels."""
    if session.npz_path is None:
        raise ValueError("missing npz_path")
    npz = np.load(session.npz_path, allow_pickle=True)
    keys = list(npz.files)
    bin_key = choose_cebra_bin_vectors_key(keys, opts)
    if bin_key is not None:
        z = np.asarray(npz[bin_key], dtype=float)
        if z.ndim == 2:
            X = z
            y = np.arange(1, z.shape[0] + 1)
        elif z.ndim == 3:
            n_runs, n_bins, n_dim = z.shape
            X = z.reshape(n_runs * n_bins, n_dim)
            y = np.tile(np.arange(1, n_bins + 1), n_runs)
        else:
            raise ValueError(f"{bin_key} must be 2D or 3D, got {z.shape}")
        embedding_key, label_key = bin_key, "bins"
    else:
        embedding_key, label_key = choose_cebra_sample_keys(keys, opts)
        X = np.asarray(npz[embedding_key], dtype=float)
        y = np.asarray(npz[label_key]).reshape(-1)
        keep = y != 0
        X, y = X[keep], y[keep]

    return {
        "session_key": session.key,
        "animal_id": session.rat_name,
        "session_id": session.session_name or session.date_str,
        "date_str": session.date_str,
        "X": X,
        "y": y.astype(int),
        "embedding_key": embedding_key,
        "label_key": label_key,
    }


def choose_cebra_bin_vectors_key(keys: Sequence[str], opts: TrialBinDecodeOptions) -> Optional[str]:
    if opts.cebra_bin_vectors_key:
        return opts.cebra_bin_vectors_key
    for key in ("zB_runs", "zA_runs", "bin_vectors", "binVectors"):
        if key in keys:
            return key
    return None


def choose_cebra_sample_keys(keys: Sequence[str], opts: TrialBinDecodeOptions) -> Tuple[str, str]:
    if opts.cebra_embedding_key and opts.cebra_label_key:
        return opts.cebra_embedding_key, opts.cebra_label_key
    preferred = [
        ("emb_b_external", "labels_b1"),
        ("embedding", "labels"),
        ("embeddings", "labels"),
        ("emb_b_holdout", "labels_b_holdout"),
        ("emb_b_train", "labels_b_train"),
    ]
    for emb_key, label_key in preferred:
        if emb_key in keys and label_key in keys:
            return emb_key, label_key
    raise ValueError(f"could not infer CEBRA embedding/label keys from {keys}")


def build_ordered_session_pairs(sessions: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """All ordered train/test pairs from different animals."""
    pairs = []
    for train in sessions:
        for test in sessions:
            if train["session_key"] == test["session_key"]:
                continue
            if train["animal_id"] == test["animal_id"]:
                continue
            pairs.append(
                {
                    "pair": f"{train['session_key']}__to__{test['session_key']}",
                    "train_session": train["session_key"],
                    "test_session": test["session_key"],
                    "train_animal": train["animal_id"],
                    "test_animal": test["animal_id"],
                }
            )
    return pairs


def classifier_factory(opts: TrialBinDecodeOptions) -> Any:
    if opts.classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, multi_class="auto", class_weight="balanced")
    elif opts.classifier == "svm":
        clf = LinearSVC(class_weight="balanced", max_iter=10000)
    else:
        raise ValueError(f"unknown classifier: {opts.classifier}")
    return make_pipeline(StandardScaler(), clf)


def decode_ordered_pairs(
    sessions: Sequence[Mapping[str, Any]],
    ordered_pairs: Sequence[Mapping[str, str]],
    opts: TrialBinDecodeOptions,
    rng: np.random.Generator,
    feature_family: str,
) -> List[Dict[str, Any]]:
    by_key = {s["session_key"]: s for s in sessions}
    rows = []
    for pair in ordered_pairs:
        train = by_key[pair["train_session"]]
        test = by_key[pair["test_session"]]
        real = fit_predict_accuracy(train["X"], train["y"], test["X"], test["y"], opts)
        shuff_acc = np.full(opts.n_shuffles, np.nan)
        shuff_bal = np.full(opts.n_shuffles, np.nan)
        for idx in range(opts.n_shuffles):
            y_shuffle = rng.permutation(train["y"])
            acc, bal = fit_predict_accuracy(train["X"], y_shuffle, test["X"], test["y"], opts)
            shuff_acc[idx] = acc
            shuff_bal[idx] = bal
        rows.append(
            {
                **pair,
                "feature_family": feature_family,
                "accuracy": real[0],
                "balanced_accuracy": real[1],
                "shuffle_accuracy_mean": np.nanmean(shuff_acc),
                "shuffle_accuracy_std": np.nanstd(shuff_acc, ddof=1),
                "shuffle_balanced_accuracy_mean": np.nanmean(shuff_bal),
                "shuffle_balanced_accuracy_std": np.nanstd(shuff_bal, ddof=1),
                "accuracy_effect_over_shuffle": effect_over_shuffle(real[0], shuff_acc),
                "balanced_accuracy_effect_over_shuffle": effect_over_shuffle(real[1], shuff_bal),
                "accuracy_shuffle_p": empirical_upper_p(real[0], shuff_acc),
                "balanced_accuracy_shuffle_p": empirical_upper_p(real[1], shuff_bal),
            }
        )
    return rows


def load_real_cebra_decoder_summary(
    summary_csv: Path,
    ordered_pairs: Sequence[Mapping[str, str]],
    opts: TrialBinDecodeOptions,
) -> List[Dict[str, Any]]:
    """Load real source-rat -> target-rat CEBRA decoder results.

    These results come from the dedicated cross-rat latent decoder pipeline.
    They are rat-pair summaries, so they are matched to the raw-summary decoder
    by source rat and target rat. This keeps the comparison honest: the raw side
    is still a neuron-identity-free population-summary decoder, while the CEBRA
    side uses the real cross-animal latent decoder output.
    """
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    by_rat_pair: Dict[Tuple[str, str], Dict[str, str]] = {}
    with summary_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("row_type") != "pair":
                continue
            if row.get("task_scheme") != opts.cebra_decoder_task_scheme:
                continue
            if int(float(row.get("dim", "nan"))) != int(opts.cebra_decoder_dim):
                continue
            if opts.cebra_decoder_comparison is not None and row.get("comparison") != opts.cebra_decoder_comparison:
                continue
            if string_to_bool(row.get("is_diagonal")):
                continue
            by_rat_pair[(row["train_rat"], row["test_rat"])] = row

    rows: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str]] = []
    for pair in ordered_pairs:
        rat_pair = (pair["train_animal"], pair["test_animal"])
        source = by_rat_pair.get(rat_pair)
        if source is None:
            missing.append(rat_pair)
            continue

        shuffle_accuracy_std = sem_to_sd(source.get("shuffle_accuracy_sem"), source.get("n_shuffle_observations"))
        shuffle_balanced_std = sem_to_sd(source.get("shuffle_balanced_accuracy_sem"), source.get("n_shuffle_observations"))
        accuracy = parse_float(source.get("real_accuracy_mean"))
        balanced_accuracy = parse_float(source.get("real_balanced_accuracy_mean"))
        shuffle_accuracy_mean = parse_float(source.get("shuffle_accuracy_mean"))
        shuffle_balanced_mean = parse_float(source.get("shuffle_balanced_accuracy_mean"))

        rows.append(
            {
                **pair,
                "feature_family": "cebra_real_decoder",
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "shuffle_accuracy_mean": shuffle_accuracy_mean,
                "shuffle_accuracy_std": shuffle_accuracy_std,
                "shuffle_balanced_accuracy_mean": shuffle_balanced_mean,
                "shuffle_balanced_accuracy_std": shuffle_balanced_std,
                "accuracy_effect_over_shuffle": safe_effect_from_summary(
                    accuracy, shuffle_accuracy_mean, shuffle_accuracy_std
                ),
                "balanced_accuracy_effect_over_shuffle": safe_effect_from_summary(
                    balanced_accuracy, shuffle_balanced_mean, shuffle_balanced_std
                ),
                "accuracy_shuffle_p": np.nan,
                "balanced_accuracy_shuffle_p": np.nan,
                "cebra_decoder_source_csv": str(summary_csv),
                "cebra_decoder_task_scheme": opts.cebra_decoder_task_scheme,
                "cebra_decoder_dim": opts.cebra_decoder_dim,
                "cebra_decoder_comparison": opts.cebra_decoder_comparison or "",
                "cebra_decoder_n_real_observations": parse_float(source.get("n_real_observations")),
                "cebra_decoder_n_shuffle_observations": parse_float(source.get("n_shuffle_observations")),
            }
        )

    if missing:
        warnings.warn(
            "Missing CEBRA decoder summary rows for rat pairs: "
            + ", ".join(f"{a}->{b}" for a, b in missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    if not rows:
        raise ValueError(
            f"No matching CEBRA decoder rows found in {summary_csv} for "
            f"{opts.cebra_decoder_task_scheme} dim {opts.cebra_decoder_dim}"
            + (f" comparison {opts.cebra_decoder_comparison}." if opts.cebra_decoder_comparison else ".")
        )
    print(
        f"  Loaded real CEBRA decoder summary: {len(rows)} ordered pairs "
        f"({opts.cebra_decoder_task_scheme}, dim={opts.cebra_decoder_dim}"
        + (f", comparison={opts.cebra_decoder_comparison})" if opts.cebra_decoder_comparison else ")")
    )
    return rows


def fit_predict_accuracy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, opts: TrialBinDecodeOptions) -> Tuple[float, float]:
    clf = classifier_factory(opts)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred)), float(balanced_accuracy_score(y_test, pred))


def compute_session_trace_and_distribution_similarity(
    raw_sessions: Sequence[Mapping[str, Any]],
    ordered_pairs: Sequence[Mapping[str, str]],
    opts: TrialBinDecodeOptions,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    by_key = {s["session_key"]: s for s in raw_sessions}
    rows = []
    for pair in ordered_pairs:
        a = by_key[pair["train_session"]]
        b = by_key[pair["test_session"]]
        mean_a = zscore_1d(a["trial_averaged_population_mean_trace"])
        mean_b = zscore_1d(b["trial_averaged_population_mean_trace"])
        trial_a = np.asarray(a["trial_trace_population_mean"], dtype=float)
        trial_b = np.asarray(b["trial_trace_population_mean"], dtype=float)
        dist = cdist(zscore_rows(trial_a), zscore_rows(trial_b), metric="euclidean")
        shuff = shuffled_trace_similarity(mean_a, mean_b, opts.n_shuffles, rng)
        rows.append(
            {
                **pair,
                "trial_averaged_trace_correlation": safe_pearson(mean_a, mean_b),
                "trial_averaged_trace_euclidean": float(np.linalg.norm(mean_a - mean_b)),
                "trial_trace_avg_nearest_neighbor_distance": float(0.5 * (np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0)))),
                "trial_trace_wasserstein_mean_binwise": mean_binwise_wasserstein(trial_a, trial_b),
                "shuffle_trace_correlation_mean": np.nanmean(shuff),
                "shuffle_trace_correlation_std": np.nanstd(shuff, ddof=1),
                "trace_correlation_effect_over_shuffle": effect_over_shuffle(safe_pearson(mean_a, mean_b), shuff),
                "trace_correlation_shuffle_p": empirical_upper_p(safe_pearson(mean_a, mean_b), shuff),
            }
        )
    return rows


def compute_within_session_controls(
    raw_sessions: Sequence[Mapping[str, Any]],
    cebra_sessions: Sequence[Mapping[str, Any]],
    opts: TrialBinDecodeOptions,
) -> List[Dict[str, Any]]:
    cebra_by_key = {s["session_key"]: s for s in cebra_sessions}
    rows = []
    for raw in raw_sessions:
        cebra = cebra_by_key.get(raw["session_key"])
        if cebra is None:
            continue
        raw_acc, raw_bal = cross_validated_accuracy(raw["X"], raw["y"], opts)
        cebra_acc, cebra_bal = cross_validated_accuracy(cebra["X"], cebra["y"], opts)
        rows.append(
            {
                "session": raw["session_key"],
                "animal_id": raw["animal_id"],
                "raw_within_accuracy": raw_acc,
                "raw_within_balanced_accuracy": raw_bal,
                "cebra_within_accuracy": cebra_acc,
                "cebra_within_balanced_accuracy": cebra_bal,
            }
        )
    return rows


def cross_validated_accuracy(X: np.ndarray, y: np.ndarray, opts: TrialBinDecodeOptions) -> Tuple[float, float]:
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(5, np.min(counts)))
    if n_splits < 2:
        return np.nan, np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=opts.random_seed)
    clf = classifier_factory(opts)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    bal = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    return float(np.nanmean(acc)), float(np.nanmean(bal))


def paired_decode_statistics(raw_rows: Sequence[Mapping[str, Any]], cebra_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    raw_by_pair = {r["pair"]: r for r in raw_rows}
    cebra_by_pair = {r["pair"]: r for r in cebra_rows}
    shared = sorted(set(raw_by_pair) & set(cebra_by_pair))
    stats = {"shared_pairs": shared}
    for metric in ["accuracy", "balanced_accuracy", "accuracy_effect_over_shuffle", "balanced_accuracy_effect_over_shuffle"]:
        raw = np.asarray([raw_by_pair[p][metric] for p in shared], dtype=float)
        cebra = np.asarray([cebra_by_pair[p][metric] for p in shared], dtype=float)
        stats[metric] = paired_stats(raw, cebra)
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
        "n_ordered_pairs": int(len(raw)),
    }


def mean_binwise_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    n_bins = min(a.shape[1], b.shape[1])
    distances = [wasserstein_distance(a[:, idx], b[:, idx]) for idx in range(n_bins)]
    return float(np.nanmean(distances))


def shuffled_trace_similarity(a: np.ndarray, b: np.ndarray, n_shuffles: int, rng: np.random.Generator) -> np.ndarray:
    out = np.full(n_shuffles, np.nan)
    for idx in range(n_shuffles):
        out[idx] = safe_pearson(rng.permutation(a), rng.permutation(b))
    return out


def zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (values - np.nanmean(values)) / sd


def zscore_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values, axis=1, keepdims=True)
    sd = np.nanstd(values, axis=1, keepdims=True)
    sd[(~np.isfinite(sd)) | (sd == 0)] = 1.0
    return (values - mu) / sd


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 2 or np.nanstd(a[valid]) == 0 or np.nanstd(b[valid]) == 0:
        return np.nan
    return float(pearsonr(a[valid], b[valid]).statistic)


def parse_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def string_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sem_to_sd(sem_value: Any, n_value: Any) -> float:
    sem_float = parse_float(sem_value)
    n_float = parse_float(n_value)
    if not np.isfinite(sem_float) or not np.isfinite(n_float) or n_float <= 0:
        return np.nan
    return float(sem_float * math.sqrt(n_float))


def safe_effect_from_summary(real: float, shuffle_mean: float, shuffle_sd: float) -> float:
    if not np.isfinite(real) or not np.isfinite(shuffle_mean) or not np.isfinite(shuffle_sd) or shuffle_sd == 0:
        return np.nan
    return float((real - shuffle_mean) / shuffle_sd)


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


def save_outputs(results: Mapping[str, Any], opts: TrialBinDecodeOptions) -> None:
    write_csv(opts.output_dir / "raw_summary_cross_animal_decoding.csv", results["raw_decode"])
    write_csv(opts.output_dir / "cebra_cross_animal_decoding.csv", results["cebra_decode"])
    write_csv(opts.output_dir / "session_trace_similarity.csv", results["session_similarity"])
    write_csv(opts.output_dir / "within_session_decoding.csv", results["within_session"])
    write_stats_csv(opts.output_dir / "trial_bin_population_summary_vs_cebra_decoding_summary.csv", results["stats"])
    np.savez_compressed(
        opts.output_dir / "trial_bin_population_summary_vs_cebra_decoding_results.npz",
        raw_session_keys=np.asarray([s["session_key"] for s in results["raw_sessions"]], dtype=object),
        raw_summary_features=np.asarray([s["X"] for s in results["raw_sessions"]], dtype=object),
        raw_summary_labels=np.asarray([s["y"] for s in results["raw_sessions"]], dtype=object),
        cebra_session_keys=np.asarray([s["session_key"] for s in results["cebra_sessions"]], dtype=object),
        cebra_features=np.asarray([s["X"] for s in results["cebra_sessions"]], dtype=object),
        cebra_labels=np.asarray([s["y"] for s in results["cebra_sessions"]], dtype=object),
        raw_decode=np.asarray(results["raw_decode"], dtype=object),
        cebra_decode=np.asarray(results["cebra_decode"], dtype=object),
        session_similarity=np.asarray(results["session_similarity"], dtype=object),
        within_session=np.asarray(results["within_session"], dtype=object),
        stats=np.asarray(results["stats"], dtype=object),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def plot_trial_bin_decoding_results(results: Mapping[str, Any], opts: TrialBinDecodeOptions) -> None:
    sns.set_theme(style="white", context="paper", font="Arial")
    raw_sessions = results["raw_sessions"]
    raw_decode = results["raw_decode"]
    cebra_decode = results["cebra_decode"]
    within = results["within_session"]

    fig = plt.figure(figsize=(12.0, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_trial_averaged_traces(ax_a, raw_sessions)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_single_trial_traces(ax_b, raw_sessions)

    ax_c = fig.add_subplot(gs[0, 2])
    plot_cross_animal_decoding(ax_c, raw_decode, cebra_decode, metric="balanced_accuracy")

    ax_d = fig.add_subplot(gs[1, 0])
    plot_cross_animal_decoding(ax_d, raw_decode, cebra_decode, metric="balanced_accuracy_effect_over_shuffle", chance_line=False)
    ax_d.set_ylabel("(real - shuffle mean) / shuffle SD")
    ax_d.set_title("D  Effect size over shuffle", loc="left")

    ax_e = fig.add_subplot(gs[1, 1])
    plot_within_session(ax_e, within)

    ax_f = fig.add_subplot(gs[1, 2])
    plot_trial_distribution_similarity(ax_f, results["session_similarity"])

    if opts.cebra_decoder_summary_csv is None:
        title = "Single-trial population-summary calcium features vs CEBRA latent features"
    else:
        title = "Single-trial population-summary calcium features vs real cross-rat CEBRA decoder"
    fig.suptitle(title, fontsize=13)
    fig.savefig(opts.output_dir / "trial_bin_population_summary_vs_cebra_decoding_figure.png", dpi=300)
    fig.savefig(opts.output_dir / "trial_bin_population_summary_vs_cebra_decoding_figure.svg")
    plt.close(fig)


def plot_trial_averaged_traces(ax: Any, sessions: Sequence[Mapping[str, Any]]) -> None:
    for session in sessions:
        y = zscore_1d(session["trial_averaged_population_mean_trace"])
        ax.plot(np.arange(1, len(y) + 1), y, "-o", linewidth=1.2, markersize=4, label=session["animal_id"])
    ax.set_title("A  Trial-averaged population calcium traces", loc="left")
    ax.set_xlabel("CSUS5 task bin")
    ax.set_ylabel("Z-scored population mean")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)


def plot_single_trial_traces(ax: Any, sessions: Sequence[Mapping[str, Any]]) -> None:
    colors = sns.color_palette("tab10", n_colors=len(sessions))
    for color, session in zip(colors, sessions):
        traces = np.asarray(session["trial_trace_population_mean"], dtype=float)
        traces_z = zscore_rows(traces)
        x = np.arange(1, traces_z.shape[1] + 1)
        for row in traces_z:
            ax.plot(x, row, color=color, alpha=0.08, linewidth=0.5)
        ax.plot(x, np.nanmean(traces_z, axis=0), color=color, linewidth=1.8, label=session["animal_id"])
    ax.set_title("B  Single-trial population calcium traces", loc="left")
    ax.set_xlabel("CSUS5 task bin")
    ax.set_ylabel("Within-trial z-scored population mean")
    sns.despine(ax=ax)


def plot_cross_animal_decoding(
    ax: Any,
    raw_rows: Sequence[Mapping[str, Any]],
    cebra_rows: Sequence[Mapping[str, Any]],
    metric: str,
    chance_line: bool = True,
) -> None:
    raw_by_pair = {r["pair"]: r for r in raw_rows}
    cebra_by_pair = {r["pair"]: r for r in cebra_rows}
    shared = sorted(set(raw_by_pair) & set(cebra_by_pair))
    raw = np.asarray([raw_by_pair[p][metric] for p in shared], dtype=float)
    cebra = np.asarray([cebra_by_pair[p][metric] for p in shared], dtype=float)
    cebra_label = "CEBRA latent features"
    if cebra_rows and cebra_rows[0].get("feature_family") == "cebra_real_decoder":
        cebra_label = "CEBRA latent decoder"
    for rv, cv in zip(raw, cebra):
        ax.plot([0, 1], [rv, cv], color="0.75", linewidth=0.7, zorder=0)
    ax.scatter(np.zeros_like(raw), raw, color="#4C78A8", edgecolor="white", linewidth=0.4, s=22, label="Population-summary calcium features")
    ax.scatter(np.ones_like(cebra), cebra, color="#E45756", edgecolor="white", linewidth=0.4, s=22, label=cebra_label)
    ax.errorbar([0, 1], [np.nanmean(raw), np.nanmean(cebra)], yerr=[sem(raw), sem(cebra)], fmt="ks", capsize=4, markersize=5)
    if chance_line:
        ax.axhline(0.20, color="0.45", linestyle="--", linewidth=1)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("C  Cross-animal CSUS5 decoding", loc="left")
    ax.set_xticks([0, 1], ["Population\nsummary", "CEBRA\nlatent"])
    sns.despine(ax=ax)


def plot_within_session(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        ax.axis("off")
        return
    raw = np.asarray([r["raw_within_balanced_accuracy"] for r in rows], dtype=float)
    cebra = np.asarray([r["cebra_within_balanced_accuracy"] for r in rows], dtype=float)
    for rv, cv in zip(raw, cebra):
        ax.plot([0, 1], [rv, cv], color="0.75", linewidth=0.8)
    ax.scatter(np.zeros_like(raw), raw, color="#4C78A8", edgecolor="white", linewidth=0.4, s=28)
    ax.scatter(np.ones_like(cebra), cebra, color="#E45756", edgecolor="white", linewidth=0.4, s=28)
    ax.errorbar([0, 1], [np.nanmean(raw), np.nanmean(cebra)], yerr=[sem(raw), sem(cebra)], fmt="ks", capsize=4, markersize=5)
    ax.axhline(0.20, color="0.45", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["Population\nsummary", "CEBRA\nlatent"])
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("E  Within-session decoding", loc="left")
    sns.despine(ax=ax)


def plot_trial_distribution_similarity(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    vals = np.asarray([r["trial_trace_avg_nearest_neighbor_distance"] for r in rows], dtype=float)
    ws = np.asarray([r["trial_trace_wasserstein_mean_binwise"] for r in rows], dtype=float)
    ax.scatter(vals, ws, color="#72B7B2", edgecolor="white", linewidth=0.4, s=28)
    ax.set_xlabel("Avg nearest-neighbor distance")
    ax.set_ylabel("Mean binwise Wasserstein")
    ax.set_title("F  Trial-distribution similarity", loc="left")
    sns.despine(ax=ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("trial_bin_population_summary_vs_cebra_decoding_output"))
    parser.add_argument("--classifier", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--active-use-zscore", action="store_true")
    parser.add_argument(
        "--baseline-subtract-raw",
        action="store_true",
        help="Subtract each trial x neuron's pre-CS mean before computing population-summary features.",
    )
    parser.add_argument("--cebra-embedding-key", default=None)
    parser.add_argument("--cebra-label-key", default=None)
    parser.add_argument("--cebra-bin-vectors-key", default=None)
    parser.add_argument(
        "--cebra-decoder-summary-csv",
        type=Path,
        default=None,
        help="Optional real cross-rat CEBRA decoder summary CSV. When provided, this replaces local CEBRA NPZ decoding.",
    )
    parser.add_argument("--cebra-decoder-task-scheme", default="CSUS5")
    parser.add_argument("--cebra-decoder-dim", type=int, default=3)
    parser.add_argument(
        "--cebra-decoder-comparison",
        default=None,
        help="Optional comparison label in combined CEBRA summaries, e.g. A_to_A, B_to_B, or A_to_B.",
    )
    parser.add_argument("--skip-within-session-cv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = TrialBinDecodeOptions(
        classifier=args.classifier,
        n_shuffles=args.n_shuffles,
        random_seed=args.random_seed,
        min_trials=args.min_trials,
        min_neurons=args.min_neurons,
        active_threshold=args.active_threshold,
        active_use_zscore=args.active_use_zscore,
        baseline_subtract_raw=args.baseline_subtract_raw,
        output_dir=args.output_dir,
        cebra_embedding_key=args.cebra_embedding_key,
        cebra_label_key=args.cebra_label_key,
        cebra_bin_vectors_key=args.cebra_bin_vectors_key,
        cebra_decoder_summary_csv=args.cebra_decoder_summary_csv,
        cebra_decoder_task_scheme=args.cebra_decoder_task_scheme,
        cebra_decoder_dim=args.cebra_decoder_dim,
        cebra_decoder_comparison=args.cebra_decoder_comparison,
        run_within_session_cv=not args.skip_within_session_cv,
    )
    sessions = load_sessions_csv(args.session_csv)
    run_trial_bin_population_summary_vs_cebra_decoding(sessions, opts)


if __name__ == "__main__":
    main()
