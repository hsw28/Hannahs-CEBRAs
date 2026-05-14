#!/usr/bin/env python3
"""
Raw-calcium-vs-CEBRA task-geometry comparison.

This analysis compares across-animal conservation of task-state geometry in:
  A) mean calcium temporal profiles
  B) raw calcium population geometry
  C) CEBRA latent geometry

It does not compare neurons directly across rats. Instead, each rat/session
gets its own task-bin representational dissimilarity matrix (RDM), and the
upper triangles of those RDMs are compared across rats.

Expected raw data fields mirror the MATLAB organization:
  rat0222.Ca_traces.CA_traces_2023_04_18
  rat0222.Ca_ts.CA_time_2023_04_18
  rat0222.CS_times.CS_2023_04_18

CEBRA input is a full embeddings_and_labels.npz file. By default, the script
uses emb_b_external / labels_b1 when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon


@dataclass
class GeometryOptions:
    task_schemes: Tuple[str, ...] = ("CSUS5",)
    win: Tuple[float, float] = (0.0, 2.0)
    pre_cs_win: Tuple[float, float] = (-1.0, 0.0)
    csus2_edges: Tuple[float, ...] = (0.0, 0.75, 2.0)
    csus5_edges: Tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    raw_distance_metric: str = "correlation"
    cebra_distance_metric: str = "euclidean"
    zscore_mode: str = "samples"
    n_shuffles: int = 500
    random_seed: int = 1
    min_trials: int = 5
    min_neurons: int = 10
    output_dir: Path = Path("raw_vs_cebra_geometry_output")
    save_figs: bool = True
    make_supplement_events: bool = False
    demean_pre_cs: bool = True
    cebra_embedding_key: Optional[str] = None
    cebra_label_key: Optional[str] = None
    cebra_bin_vectors_key: Optional[str] = None
    max_rdms_to_plot: int = 6


@dataclass
class SessionSpec:
    rat_name: str
    date_str: str
    session_name: Optional[str] = None
    animal_mat: Optional[Path] = None
    npz_path: Optional[Path] = None

    @property
    def key(self) -> str:
        session = self.session_name or self.date_str
        return f"{self.rat_name}__{session}"


@dataclass
class Hdf5Animal:
    mat_path: Path
    rat_name: str


def run_raw_vs_cebra_geometry_comparison(
    sessions: Sequence[SessionSpec],
    animal_structs: Optional[Mapping[str, Any]] = None,
    opts: Optional[GeometryOptions] = None,
) -> Dict[str, Any]:
    """Run the full raw calcium vs CEBRA RDM comparison."""
    opts = opts or GeometryOptions()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.random_seed)

    print("\nRaw-vs-CEBRA geometry comparison")
    print(f"  Sessions listed: {len(sessions)}")
    print(f"  Output dir: {opts.output_dir}")

    results: Dict[str, Any] = {
        "opts": opts,
        "raw": {},
        "events": {},
        "cebra": {},
        "similarity": {},
        "stats": {},
        "shuffle": {},
    }

    for task_scheme in opts.task_schemes:
        print(f"\n=== Task scheme: {task_scheme} ===")
        raw_records = []
        event_records = []

        for idx, session in enumerate(sessions, start=1):
            print(f"  [{idx}/{len(sessions)}] {session.rat_name} {session.date_str}")
            try:
                animal = get_animal_data(session, animal_structs)
                calcium_all, calcium_ts, trial_cs = get_calcium_session(animal, session)
            except Exception as exc:
                warnings.warn(f"{session.key}: missing raw data ({exc}); skipping raw calcium.")
                continue

            if len(trial_cs) < opts.min_trials:
                warnings.warn(f"{session.key}: only {len(trial_cs)} CS trials; skipping.")
                continue

            align_win = (min(opts.win[0], opts.pre_cs_win[0]), max(opts.win[1], opts.pre_cs_win[1]))
            try:
                calcium_aligned, aligned_ts = extract_task_aligned_calcium(
                    calcium_all, calcium_ts, trial_cs, align_win
                )
                bin_vectors, info = compute_raw_task_bin_population_vectors(
                    calcium_aligned, aligned_ts, task_scheme, opts
                )
            except Exception as exc:
                warnings.warn(f"{session.key}: raw calcium processing failed ({exc}); skipping.")
                continue

            if bin_vectors.shape[1] < opts.min_neurons:
                warnings.warn(
                    f"{session.key}: only {bin_vectors.shape[1]} valid neurons; "
                    f"min_neurons={opts.min_neurons}; skipping."
                )
                continue

            raw_records.append(
                {
                    "name": session.key,
                    "rdm": compute_rdm(bin_vectors, opts.raw_distance_metric),
                    "bin_vectors": bin_vectors,
                    "mean_temporal_profile": info["mean_temporal_profile"],
                    "mean_temporal_profile_by_bin": info["mean_temporal_profile_by_bin"],
                    "info": info,
                }
            )

            if opts.make_supplement_events:
                try:
                    event_all = get_event_session(animal, session)
                    event_aligned, event_ts = extract_task_aligned_calcium(event_all, calcium_ts, trial_cs, align_win)
                    event_vectors, event_info = compute_raw_task_bin_population_vectors(
                        event_aligned, event_ts, task_scheme, opts
                    )
                    if event_vectors.shape[1] >= opts.min_neurons:
                        event_records.append(
                            {
                                "name": session.key,
                                "rdm": compute_rdm(event_vectors, opts.raw_distance_metric),
                                "bin_vectors": event_vectors,
                                "info": event_info,
                            }
                        )
                except Exception as exc:
                    warnings.warn(f"{session.key}: event supplement skipped ({exc}).")

        cebra_records = compute_cebra_task_bin_rdms(sessions, task_scheme, opts)
        raw_comparison = compare_rdms_across_rats(raw_records)
        cebra_comparison = compare_rdms_across_rats(cebra_records)
        profile_comparison = compare_mean_temporal_profiles(raw_records)
        event_comparison = compare_rdms_across_rats(event_records) if event_records else None
        stats = compare_raw_and_cebra(raw_comparison, cebra_comparison)
        shuffle = shuffle_rdm_comparison(raw_records, cebra_records, opts, rng)

        results["raw"][task_scheme] = raw_records
        results["events"][task_scheme] = event_records
        results["cebra"][task_scheme] = cebra_records
        results["similarity"][task_scheme] = {
            "raw": raw_comparison,
            "cebra": cebra_comparison,
            "mean_temporal_profile": profile_comparison,
            "events": event_comparison,
        }
        results["stats"][task_scheme] = stats
        results["shuffle"][task_scheme] = shuffle

        print(f"  Raw valid datasets: {len(raw_records)}")
        print(f"  CEBRA valid datasets: {len(cebra_records)}")
        if not raw_comparison["valid_correlation"]:
            warnings.warn(
                f"RDM correlations are not valid for {task_scheme}: too few task-bin distances. "
                "Use CSUS5 or more bins for the primary RDM-correlation analysis."
            )

    save_outputs(results, opts)
    if opts.save_figs:
        plot_raw_vs_cebra_geometry_results(results, opts)
    print(f"\nDone. Results saved to {opts.output_dir}")
    return results


def load_sessions_csv(path: Path) -> List[SessionSpec]:
    sessions = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sessions.append(
                SessionSpec(
                    rat_name=str(row.get("rat_name") or row.get("ratName")),
                    date_str=str(row.get("date_str") or row.get("dateStr")),
                    session_name=empty_to_none(row.get("session_name") or row.get("sessionName")),
                    animal_mat=path_or_none(row.get("animal_mat") or row.get("animalMat")),
                    npz_path=path_or_none(row.get("npz_path") or row.get("npzPath")),
                )
            )
    return sessions


def empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def path_or_none(value: Optional[str]) -> Optional[Path]:
    value = empty_to_none(value)
    return Path(value) if value else None


def get_animal_data(session: SessionSpec, animal_structs: Optional[Mapping[str, Any]]) -> Any:
    if animal_structs is not None and session.rat_name in animal_structs:
        return animal_structs[session.rat_name]
    if session.animal_mat is None:
        raise ValueError("no animal_mat path or animal_structs entry")
    try:
        mat = loadmat(session.animal_mat, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return Hdf5Animal(session.animal_mat, session.rat_name)
    if session.rat_name in mat:
        return mat[session.rat_name]
    candidates = [k for k in mat if not k.startswith("__")]
    if len(candidates) == 1:
        return mat[candidates[0]]
    raise KeyError(f"{session.rat_name} not found in {session.animal_mat}; variables: {candidates}")


def get_nested_field(obj: Any, *fields: str) -> Any:
    cur = obj
    for field_name in fields:
        if isinstance(cur, Mapping):
            cur = cur[field_name]
        else:
            cur = getattr(cur, field_name)
    return cur


def get_calcium_session(animal: Any, session: SessionSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(animal, Hdf5Animal):
        return get_hdf5_calcium_session(animal, session)
    date = session.date_str
    calcium = get_nested_field(animal, "Ca_traces", f"CA_traces_{date}")
    calcium_ts = get_nested_field(animal, "Ca_ts", f"CA_time_{date}")
    trial_cs = get_nested_field(animal, "CS_times", f"CS_{date}")
    return np.asarray(calcium), np.asarray(calcium_ts), np.asarray(trial_cs).reshape(-1)


def get_event_session(animal: Any, session: SessionSpec) -> np.ndarray:
    if isinstance(animal, Hdf5Animal):
        import h5py

        with h5py.File(animal.mat_path, "r") as handle:
            return np.asarray(handle[animal.rat_name]["Ca_peaks"][f"CA_peaks_{session.date_str}"])
    return np.asarray(get_nested_field(animal, "Ca_peaks", f"CA_peaks_{session.date_str}"))


def get_hdf5_calcium_session(animal: Hdf5Animal, session: SessionSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    date = session.date_str
    with h5py.File(animal.mat_path, "r") as handle:
        rat = handle[animal.rat_name]
        calcium = np.asarray(rat["Ca_traces"][f"CA_traces_{date}"])
        calcium_ts = np.asarray(rat["Ca_ts"][f"CA_time_{date}"])
        trial_cs = np.asarray(rat["CS_times"][f"CS_{date}"]).reshape(-1)
    return calcium, calcium_ts, trial_cs


def normalize_calcium_ts(calcium_ts: np.ndarray) -> np.ndarray:
    calcium_ts = np.asarray(calcium_ts, dtype=float)
    if calcium_ts.ndim == 2 and calcium_ts.shape[1] == 2:
        return calcium_ts[:, 1][1::2] / 1000.0
    if calcium_ts.ndim == 2 and calcium_ts.shape[0] == 2:
        return calcium_ts[1, :][1::2] / 1000.0
    if calcium_ts.ndim == 2 and calcium_ts.shape[0] >= 3:
        return calcium_ts[1, :][1::2] / 1000.0
    return calcium_ts.reshape(-1)


def orient_calcium(calcium_all: np.ndarray, calcium_ts: np.ndarray) -> np.ndarray:
    calcium_all = np.asarray(calcium_all, dtype=float)
    n_time = len(calcium_ts)
    if calcium_all.shape[0] == n_time:
        return calcium_all
    if calcium_all.shape[1] == n_time:
        return calcium_all.T
    raise ValueError(f"calcium shape {calcium_all.shape} does not match timestamp length {n_time}")


def extract_task_aligned_calcium(
    calcium_all: np.ndarray,
    calcium_ts: np.ndarray,
    trial_cs: np.ndarray,
    win: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    calcium_ts = normalize_calcium_ts(calcium_ts)
    calcium_time_by_neurons = orient_calcium(calcium_all, calcium_ts)

    unique_ts, unique_idx = np.unique(calcium_ts, return_index=True)
    calcium_time_by_neurons = calcium_time_by_neurons[unique_idx]
    dt = np.nanmedian(np.diff(unique_ts))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("could not infer positive sampling interval")

    aligned_ts = np.arange(win[0], win[1] + dt / 2.0, dt)
    aligned = np.full((len(trial_cs), len(aligned_ts), calcium_time_by_neurons.shape[1]), np.nan)
    for trial_idx, cs_time in enumerate(np.asarray(trial_cs, dtype=float).reshape(-1)):
        query = cs_time + aligned_ts
        valid = (query >= unique_ts[0]) & (query <= unique_ts[-1])
        if np.any(valid):
            for neuron_idx in range(calcium_time_by_neurons.shape[1]):
                aligned[trial_idx, valid, neuron_idx] = np.interp(
                    query[valid], unique_ts, calcium_time_by_neurons[:, neuron_idx]
                )
    valid_neurons = np.any(np.isfinite(aligned), axis=(0, 1))
    return aligned[:, :, valid_neurons], aligned_ts


def make_task_bin_labels(
    aligned_ts: np.ndarray, task_scheme: str, opts: GeometryOptions
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if task_scheme.upper() == "CSUS2":
        edges = np.asarray(opts.csus2_edges, dtype=float)
    elif task_scheme.upper() == "CSUS5":
        edges = np.asarray(opts.csus5_edges, dtype=float)
    else:
        raise ValueError(f"unknown task scheme {task_scheme}")

    labels = np.full(aligned_ts.shape, np.nan)
    for bin_idx in range(len(edges) - 1):
        if bin_idx < len(edges) - 2:
            mask = (aligned_ts >= edges[bin_idx]) & (aligned_ts < edges[bin_idx + 1])
        else:
            mask = (aligned_ts >= edges[bin_idx]) & (aligned_ts <= edges[bin_idx + 1])
        labels[mask] = bin_idx + 1
    names = [f"bin{idx + 1}_{edges[idx]:g}_to_{edges[idx + 1]:g}" for idx in range(len(edges) - 1)]
    return labels, edges, names


def zscore_samples(calcium_aligned: np.ndarray, task_time_mask: np.ndarray) -> np.ndarray:
    out = calcium_aligned.copy()
    for neuron_idx in range(out.shape[2]):
        vals = out[:, task_time_mask, neuron_idx].reshape(-1)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        out[:, :, neuron_idx] = (out[:, :, neuron_idx] - mu) / sigma
    return out


def zscore_columns(values: np.ndarray) -> np.ndarray:
    mu = np.nanmean(values, axis=0, keepdims=True)
    sigma = np.nanstd(values, axis=0, keepdims=True)
    sigma[(~np.isfinite(sigma)) | (sigma == 0)] = 1.0
    return (values - mu) / sigma


def compute_raw_task_bin_population_vectors(
    calcium_aligned: np.ndarray,
    aligned_ts: np.ndarray,
    task_scheme: str,
    opts: GeometryOptions,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bin_labels, bin_edges, bin_names = make_task_bin_labels(aligned_ts, task_scheme, opts)
    task_time_mask = np.isfinite(bin_labels)
    samples = calcium_aligned
    if opts.zscore_mode == "samples":
        samples = zscore_samples(samples, task_time_mask)

    n_bins = len(bin_edges) - 1
    bin_vectors = np.full((n_bins, calcium_aligned.shape[2]), np.nan)
    for bin_idx in range(n_bins):
        tmask = bin_labels == (bin_idx + 1)
        if np.any(tmask):
            bin_vectors[bin_idx] = np.nanmean(samples[:, tmask, :], axis=(0, 1))

    if opts.zscore_mode == "binMeans":
        bin_vectors = zscore_columns(bin_vectors)
    elif opts.zscore_mode not in {"samples", "none"}:
        raise ValueError(f"unknown zscore_mode: {opts.zscore_mode}")

    valid_neurons = np.all(np.isfinite(bin_vectors), axis=0)
    bin_vectors = bin_vectors[:, valid_neurons]

    mean_temporal_profile = np.nanmean(calcium_aligned[:, :, valid_neurons], axis=(0, 2))
    if opts.demean_pre_cs:
        pre_mask = (aligned_ts >= opts.pre_cs_win[0]) & (aligned_ts < opts.pre_cs_win[1])
        if np.any(pre_mask):
            mean_temporal_profile = mean_temporal_profile - np.nanmean(mean_temporal_profile[pre_mask])

    profile_by_bin = np.full(n_bins, np.nan)
    for bin_idx in range(n_bins):
        tmask = bin_labels == (bin_idx + 1)
        if np.any(tmask):
            profile_by_bin[bin_idx] = np.nanmean(mean_temporal_profile[tmask])

    return bin_vectors, {
        "aligned_ts": aligned_ts,
        "bin_labels": bin_labels,
        "bin_edges": bin_edges,
        "bin_names": bin_names,
        "valid_neurons": valid_neurons,
        "mean_temporal_profile": mean_temporal_profile,
        "mean_temporal_profile_by_bin": profile_by_bin,
    }


def compute_rdm(bin_vectors: np.ndarray, metric: str) -> np.ndarray:
    if bin_vectors.shape[0] < 2:
        return np.full((bin_vectors.shape[0], bin_vectors.shape[0]), np.nan)
    if metric == "correlation":
        distances = pdist(bin_vectors, metric="correlation")
    elif metric == "euclidean":
        distances = pdist(bin_vectors, metric="euclidean")
    else:
        raise ValueError(f"unknown distance metric: {metric}")
    return squareform(distances)


def load_cebra_npz(npz_path: Path, opts: GeometryOptions) -> Tuple[np.ndarray, np.ndarray, str, str]:
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.files)
    bin_key = choose_cebra_bin_vectors_key(keys, opts)
    if bin_key is not None:
        bin_vectors = np.asarray(npz[bin_key], dtype=float)
        if bin_vectors.ndim == 3:
            bin_vectors = np.nanmean(bin_vectors, axis=0)
        if bin_vectors.ndim != 2:
            raise ValueError(f"{npz_path}: {bin_key} must be 2D or 3D, got {bin_vectors.shape}")
        labels = np.asarray(npz["bins"] if "bins" in keys else np.arange(1, bin_vectors.shape[0] + 1)).reshape(-1)
        return bin_vectors, labels, bin_key, "bins"
    embedding_key, label_key = choose_cebra_keys(keys, opts)
    embedding = np.asarray(npz[embedding_key], dtype=float)
    labels = np.asarray(npz[label_key]).reshape(-1)
    if embedding.shape[0] != len(labels) and embedding.shape[1] == len(labels):
        embedding = embedding.T
    if embedding.shape[0] != len(labels):
        raise ValueError(
            f"{npz_path}: {embedding_key} has {embedding.shape[0]} rows but {label_key} has {len(labels)} labels"
        )
    return embedding, labels, embedding_key, label_key


def choose_cebra_bin_vectors_key(keys: Sequence[str], opts: GeometryOptions) -> Optional[str]:
    if opts.cebra_bin_vectors_key:
        return opts.cebra_bin_vectors_key
    for key in ("zB_runs", "zA_runs", "bin_vectors", "binVectors"):
        if key in keys:
            return key
    return None


def choose_cebra_keys(keys: Sequence[str], opts: GeometryOptions) -> Tuple[str, str]:
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
    emb_keys = [k for k in keys if k.startswith("emb") or "embedding" in k]
    label_keys = [k for k in keys if k.startswith("label") or "label" in k]
    if not emb_keys or not label_keys:
        raise ValueError(f"could not infer embedding/label keys from {keys}")
    warnings.warn(
        f"Using inferred CEBRA keys {emb_keys[0]}/{label_keys[0]}; "
        "set --cebra-embedding-key and --cebra-label-key to override."
    )
    return emb_keys[0], label_keys[0]


def compute_cebra_task_bin_rdms(
    sessions: Sequence[SessionSpec], task_scheme: str, opts: GeometryOptions
) -> List[Dict[str, Any]]:
    records = []
    for session in sessions:
        if session.npz_path is None:
            continue
        try:
            embedding, labels, emb_key, label_key = load_cebra_npz(session.npz_path, opts)
        except Exception as exc:
            warnings.warn(f"{session.key}: could not load CEBRA npz ({exc}); skipping.")
            continue
        pre_binned = emb_key in {"zB_runs", "zA_runs", "bin_vectors", "binVectors"} or (
            embedding.shape[0] == len(labels) and len(labels) <= 20
        )
        if pre_binned:
            unique_labels = np.asarray(labels)
            bin_vectors = embedding
        else:
            unique_labels = np.asarray([x for x in np.unique(labels) if x != 0])
            bin_vectors = np.vstack([np.nanmean(embedding[labels == label], axis=0) for label in unique_labels])
        if len(unique_labels) < 2:
            warnings.warn(f"{session.key}: fewer than two CEBRA labels; skipping.")
            continue

        expected_bins = len(make_task_bin_labels(np.asarray(opts.win), task_scheme, opts)[1]) - 1
        if len(unique_labels) != expected_bins:
            warnings.warn(
                f"{session.key}: CEBRA labels contain {len(unique_labels)} bins, "
                f"expected {expected_bins} for {task_scheme}."
            )
        records.append(
            {
                "name": session.key,
                "rdm": compute_rdm(bin_vectors, opts.cebra_distance_metric),
                "bin_vectors": bin_vectors,
                "labels": unique_labels,
                "npz_path": str(session.npz_path),
                "embedding_key": emb_key,
                "label_key": label_key,
            }
        )
    return records


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices_from(rdm, k=1)]


def compare_rdms_across_rats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [r["name"] for r in records]
    n = len(records)
    pearson = np.full((n, n), np.nan)
    spearman = np.full((n, n), np.nan)
    distance_difference = np.full((n, n), np.nan)
    np.fill_diagonal(pearson, 1.0)
    np.fill_diagonal(spearman, 1.0)
    np.fill_diagonal(distance_difference, 0.0)
    valid_correlation = True

    for i in range(n):
        for j in range(i + 1, n):
            v1 = upper_triangle(records[i]["rdm"])
            v2 = upper_triangle(records[j]["rdm"])
            valid = np.isfinite(v1) & np.isfinite(v2)
            v1, v2 = v1[valid], v2[valid]
            if len(v1) < 2:
                valid_correlation = False
                pr = sr = np.nan
            else:
                pr = pearsonr(v1, v2).statistic if np.nanstd(v1) > 0 and np.nanstd(v2) > 0 else np.nan
                sr = spearmanr(v1, v2).correlation
            dd = np.nanmean(np.abs(v1 - v2)) if len(v1) else np.nan
            pearson[i, j] = pearson[j, i] = pr
            spearman[i, j] = spearman[j, i] = sr
            distance_difference[i, j] = distance_difference[j, i] = dd

    offdiag = pearson[np.triu_indices(n, k=1)]
    offdiag = offdiag[np.isfinite(offdiag)]
    return {
        "names": names,
        "pearson": pearson,
        "spearman": spearman,
        "distance_difference": distance_difference,
        "valid_correlation": valid_correlation,
        "off_diagonal_pearson": offdiag,
        "mean_pearson": np.nanmean(offdiag) if len(offdiag) else np.nan,
        "sem_pearson": sem(offdiag),
    }


def compare_mean_temporal_profiles(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [r["name"] for r in records]
    profiles = [np.asarray(r["mean_temporal_profile_by_bin"], dtype=float) for r in records]
    n = len(profiles)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            valid = np.isfinite(profiles[i]) & np.isfinite(profiles[j])
            r = pearsonr(profiles[i][valid], profiles[j][valid]).statistic if np.sum(valid) >= 2 else np.nan
            matrix[i, j] = matrix[j, i] = r
    vals = matrix[np.triu_indices(n, k=1)]
    vals = vals[np.isfinite(vals)]
    return {"names": names, "pearson": matrix, "off_diagonal": vals, "mean": np.nanmean(vals) if len(vals) else np.nan}


def compare_raw_and_cebra(raw_comp: Mapping[str, Any], cebra_comp: Mapping[str, Any]) -> Dict[str, Any]:
    shared = [name for name in raw_comp["names"] if name in cebra_comp["names"]]
    raw_idx = {name: i for i, name in enumerate(raw_comp["names"])}
    cebra_idx = {name: i for i, name in enumerate(cebra_comp["names"])}
    raw_vals, cebra_vals, pair_names = [], [], []
    for a_idx, a in enumerate(shared):
        for b in shared[a_idx + 1 :]:
            rv = raw_comp["pearson"][raw_idx[a], raw_idx[b]]
            cv = cebra_comp["pearson"][cebra_idx[a], cebra_idx[b]]
            if np.isfinite(rv) and np.isfinite(cv):
                raw_vals.append(rv)
                cebra_vals.append(cv)
                pair_names.append(f"{a} vs {b}")
    raw_vals = np.asarray(raw_vals, dtype=float)
    cebra_vals = np.asarray(cebra_vals, dtype=float)
    if len(raw_vals) > 1:
        t_p = ttest_rel(raw_vals, cebra_vals, nan_policy="omit").pvalue
        try:
            w_p = wilcoxon(cebra_vals - raw_vals).pvalue
        except ValueError:
            w_p = np.nan
    else:
        t_p = w_p = np.nan
    return {
        "paired_names": pair_names,
        "paired_raw_values": raw_vals,
        "paired_cebra_values": cebra_vals,
        "mean_raw": np.nanmean(raw_vals) if len(raw_vals) else np.nan,
        "mean_cebra": np.nanmean(cebra_vals) if len(cebra_vals) else np.nan,
        "sem_raw": sem(raw_vals),
        "sem_cebra": sem(cebra_vals),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def shuffle_rdm_comparison(
    raw_records: Sequence[Mapping[str, Any]],
    cebra_records: Sequence[Mapping[str, Any]],
    opts: GeometryOptions,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    raw_actual = compare_rdms_across_rats(raw_records)["mean_pearson"]
    cebra_actual = compare_rdms_across_rats(cebra_records)["mean_pearson"]
    raw_null = np.full(opts.n_shuffles, np.nan)
    cebra_null = np.full(opts.n_shuffles, np.nan)
    for idx in range(opts.n_shuffles):
        raw_null[idx] = compare_rdms_across_rats(permute_record_rdms(raw_records, rng))["mean_pearson"]
        cebra_null[idx] = compare_rdms_across_rats(permute_record_rdms(cebra_records, rng))["mean_pearson"]
    return {
        "raw_actual_mean": raw_actual,
        "cebra_actual_mean": cebra_actual,
        "raw_mean_similarity": raw_null,
        "cebra_mean_similarity": cebra_null,
        "raw_empirical_p": empirical_upper_p(raw_actual, raw_null),
        "cebra_empirical_p": empirical_upper_p(cebra_actual, cebra_null),
    }


def permute_record_rdms(records: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        rdm = np.asarray(record["rdm"])
        perm = rng.permutation(rdm.shape[0])
        copied = dict(record)
        copied["rdm"] = rdm[perm][:, perm]
        out.append(copied)
    return out


def empirical_upper_p(actual: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(actual) or len(null) == 0:
        return np.nan
    return (np.sum(null >= actual) + 1) / (len(null) + 1)


def sem(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def save_outputs(results: Mapping[str, Any], opts: GeometryOptions) -> None:
    primary = opts.task_schemes[0]
    sim = results["similarity"][primary]
    write_matrix_csv(opts.output_dir / "raw_rdm_similarity_matrix.csv", sim["raw"]["names"], sim["raw"]["pearson"])
    write_matrix_csv(opts.output_dir / "cebra_rdm_similarity_matrix.csv", sim["cebra"]["names"], sim["cebra"]["pearson"])
    write_matrix_csv(
        opts.output_dir / "mean_temporal_profile_similarity_matrix.csv",
        sim["mean_temporal_profile"]["names"],
        sim["mean_temporal_profile"]["pearson"],
    )
    write_summary_csv(opts.output_dir / "raw_vs_cebra_geometry_summary.csv", results, opts)
    np.savez_compressed(opts.output_dir / "raw_vs_cebra_geometry_results.npz", **flatten_results_for_npz(results))


def write_matrix_csv(path: Path, names: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])


def write_summary_csv(path: Path, results: Mapping[str, Any], opts: GeometryOptions) -> None:
    rows = []
    for scheme in opts.task_schemes:
        stats = results["stats"][scheme]
        shuff = results["shuffle"][scheme]
        rows.extend(
            [
                {"task_scheme": scheme, "metric": "raw_rdm_similarity", "mean": stats["mean_raw"], "sem": stats["sem_raw"], "p_value": np.nan},
                {
                    "task_scheme": scheme,
                    "metric": "cebra_rdm_similarity",
                    "mean": stats["mean_cebra"],
                    "sem": stats["sem_cebra"],
                    "p_value": stats["paired_t_p"],
                    "wilcoxon_p": stats["wilcoxon_p"],
                },
                {
                    "task_scheme": scheme,
                    "metric": "raw_shuffle_mean_similarity",
                    "mean": np.nanmean(shuff["raw_mean_similarity"]),
                    "sem": sem(shuff["raw_mean_similarity"]),
                    "p_value": shuff["raw_empirical_p"],
                },
                {
                    "task_scheme": scheme,
                    "metric": "cebra_shuffle_mean_similarity",
                    "mean": np.nanmean(shuff["cebra_mean_similarity"]),
                    "sem": sem(shuff["cebra_mean_similarity"]),
                    "p_value": shuff["cebra_empirical_p"],
                },
            ]
        )
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_results_for_npz(results: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for scheme, sim in results["similarity"].items():
        flat[f"{scheme}_raw_similarity"] = sim["raw"]["pearson"]
        flat[f"{scheme}_cebra_similarity"] = sim["cebra"]["pearson"]
        flat[f"{scheme}_mean_temporal_profile_similarity"] = sim["mean_temporal_profile"]["pearson"]
        flat[f"{scheme}_raw_names"] = np.asarray(sim["raw"]["names"], dtype=object)
        flat[f"{scheme}_cebra_names"] = np.asarray(sim["cebra"]["names"], dtype=object)
        flat[f"{scheme}_raw_shuffle"] = results["shuffle"][scheme]["raw_mean_similarity"]
        flat[f"{scheme}_cebra_shuffle"] = results["shuffle"][scheme]["cebra_mean_similarity"]
    return flat


def plot_raw_vs_cebra_geometry_results(results: Mapping[str, Any], opts: GeometryOptions) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        warnings.warn(f"matplotlib is not installed; skipping figure output ({exc}).")
        return

    scheme = opts.task_schemes[0]
    raw_records = results["raw"][scheme]
    cebra_records = results["cebra"][scheme]
    stats = results["stats"][scheme]
    shuff = results["shuffle"][scheme]

    n_raw = min(len(raw_records), opts.max_rdms_to_plot)
    n_cebra = min(len(cebra_records), opts.max_rdms_to_plot)
    n_cols = max(4, n_raw, n_cebra)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)

    ax = fig.add_subplot(3, n_cols, (1, max(1, n_cols // 2)))
    for record in raw_records:
        ax.plot(record["mean_temporal_profile_by_bin"], marker="o", linewidth=1)
    ax.set_title(f"A  Mean temporal profiles ({scheme})")
    ax.set_xlabel("Task bin")
    ax.set_ylabel("Mean calcium")

    ax = fig.add_subplot(3, n_cols, (max(1, n_cols // 2) + 1, n_cols))
    raw_vals = np.asarray(stats["paired_raw_values"], dtype=float)
    cebra_vals = np.asarray(stats["paired_cebra_values"], dtype=float)
    for rv, cv in zip(raw_vals, cebra_vals):
        ax.plot([1, 2], [rv, cv], color="0.65", linewidth=0.8)
        ax.scatter([1], [rv], color="#1f5aa6", zorder=3)
        ax.scatter([2], [cv], color="#c8372d", zorder=3)
    jitter = np.random.default_rng(0)
    for x, vals in [(0.72, shuff["raw_mean_similarity"]), (2.28, shuff["cebra_mean_similarity"])]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.scatter(x + jitter.uniform(-0.06, 0.06, len(vals)), vals, s=6, color="0.75", alpha=0.7)
    ax.errorbar([1, 2], [np.nanmean(raw_vals), np.nanmean(cebra_vals)], yerr=[sem(raw_vals), sem(cebra_vals)], fmt="ks", capsize=4)
    ax.set_xticks([1, 2], ["Raw calcium RDM", "CEBRA RDM"])
    ax.set_ylabel("Across-rat RDM similarity (Pearson r)")
    ax.set_title(f"D  Geometry conservation  t p={stats['paired_t_p']:.3g}, W p={stats['wilcoxon_p']:.3g}")

    for i in range(n_raw):
        ax = fig.add_subplot(3, n_cols, n_cols + i + 1)
        im = ax.imshow(raw_records[i]["rdm"], aspect="equal")
        ax.set_title(f"B{i + 1} Raw {raw_records[i]['name'][:18]}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    for i in range(n_cebra):
        ax = fig.add_subplot(3, n_cols, 2 * n_cols + i + 1)
        im = ax.imshow(cebra_records[i]["rdm"], aspect="equal")
        ax.set_title(f"C{i + 1} CEBRA {cebra_records[i]['name'][:18]}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Raw calcium vs CEBRA task-state geometry comparison: {scheme}")
    fig.savefig(opts.output_dir / "raw_vs_cebra_geometry_figure.png", dpi=300)
    fig.savefig(opts.output_dir / "raw_vs_cebra_geometry_figure.svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-csv", required=True, type=Path, help="CSV with rat_name,date_str,animal_mat,npz_path columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("raw_vs_cebra_geometry_output"))
    parser.add_argument("--task-scheme", action="append", choices=["CSUS2", "CSUS5"], help="Can be passed more than once. Default: CSUS5.")
    parser.add_argument("--raw-distance-metric", choices=["correlation", "euclidean"], default="correlation")
    parser.add_argument("--cebra-distance-metric", choices=["correlation", "euclidean"], default="euclidean")
    parser.add_argument("--zscore-mode", choices=["samples", "binMeans", "none"], default="samples")
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--cebra-embedding-key", default=None)
    parser.add_argument("--cebra-label-key", default=None)
    parser.add_argument("--cebra-bin-vectors-key", default=None, help="For pre-binned geometry NPZ files, e.g. zB_runs or zA_runs.")
    parser.add_argument("--events", action="store_true", help="Also run supplement using Ca_peaks.")
    parser.add_argument("--no-figs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = GeometryOptions(
        task_schemes=tuple(args.task_scheme or ["CSUS5"]),
        raw_distance_metric=args.raw_distance_metric,
        cebra_distance_metric=args.cebra_distance_metric,
        zscore_mode=args.zscore_mode,
        n_shuffles=args.n_shuffles,
        random_seed=args.random_seed,
        min_trials=args.min_trials,
        min_neurons=args.min_neurons,
        output_dir=args.output_dir,
        save_figs=not args.no_figs,
        make_supplement_events=args.events,
        cebra_embedding_key=args.cebra_embedding_key,
        cebra_label_key=args.cebra_label_key,
        cebra_bin_vectors_key=args.cebra_bin_vectors_key,
    )
    sessions = load_sessions_csv(args.session_csv)
    run_raw_vs_cebra_geometry_comparison(sessions, opts=opts)


if __name__ == "__main__":
    main()
