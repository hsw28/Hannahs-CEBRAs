#!/usr/bin/env python3
"""Across-animal CEBRA latent-space task-epoch decoding.

This analysis asks whether task epoch boundaries learned by a decoder in one
animal's CEBRA latent space transfer to another animal's aligned latent space.
It deliberately does not match neurons across animals: each rat keeps its own
neural feature matrix, and transfer is evaluated only after embedding and
latent-space alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The desktop sandbox cannot write to /Users/Hannah/.matplotlib, so give
# matplotlib a writable cache path before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hannahs_cebras_matplotlib")
# PyArrow can emit sandbox-related CPU feature warnings when pandas imports it.
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ttest_rel, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


plt.rcParams["svg.fonttype"] = "none"


EMBEDDING_KEY_CANDIDATES = ("embedding", "embeddings", "z", "latent", "output", "model_output")
LABEL_KEY_CANDIDATES = ("labels", "label", "CSUS2", "CSUS5", "training", "y")
TRIAL_KEY_CANDIDATES = ("trial_ids", "trial_id", "trials", "trial")


@dataclass
class ZScoreTransform:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ProcrustesTransform:
    rotation: np.ndarray
    source_center: np.ndarray
    target_center: np.ndarray
    procrustes_scale_stat: float
    allow_scale: bool = False
    scale_factor: float = 1.0


def _as_2d_embedding(embeddings: Any, labels: Optional[np.ndarray] = None) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=float)
    emb = np.squeeze(emb)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D samples x dims array; got shape {emb.shape}.")
    if labels is not None and emb.shape[0] != len(labels) and emb.shape[1] == len(labels):
        emb = emb.T
    return emb


def _label_not_nan_mask(labels: np.ndarray) -> np.ndarray:
    if np.issubdtype(labels.dtype, np.number):
        return np.isfinite(labels.astype(float))
    cleaned = np.array([str(x).strip().lower() for x in labels], dtype=object)
    return np.array([(x not in {"", "nan", "none", "<na>"}) for x in cleaned], dtype=bool)


def clean_labels_and_embeddings(
    embeddings: Any, labels: Any, trial_ids: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Remove NaN labels and nonfinite latent rows.

    Parameters
    ----------
    embeddings
        Latent coordinates as samples x latent dimensions. If the array is
        dims x samples and label length matches the second axis, it is
        transposed automatically.
    labels
        CSUS2 or CSUS5 task-epoch label for each sample/time bin.
    trial_ids
        Optional trial identifier for each sample. If present, all alignment
        and decoding splits are done by trial rather than by individual bins.
    """
    labels_arr = np.asarray(labels).squeeze()
    if labels_arr.ndim != 1:
        raise ValueError(f"Labels must be one-dimensional after squeezing; got shape {labels_arr.shape}.")

    emb = _as_2d_embedding(embeddings, labels_arr)
    if emb.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            "Embedding and label lengths do not match after orientation check: "
            f"{emb.shape[0]} samples vs {labels_arr.shape[0]} labels."
        )

    finite_embedding = np.all(np.isfinite(emb), axis=1)
    valid_labels = _label_not_nan_mask(labels_arr)
    keep = finite_embedding & valid_labels

    clean_trials = None
    if trial_ids is not None:
        clean_trials = np.asarray(trial_ids).squeeze()
        if clean_trials.ndim != 1 or clean_trials.shape[0] != labels_arr.shape[0]:
            raise ValueError(
                "trial_ids must be one-dimensional and match label length; "
                f"got shape {clean_trials.shape} for {labels_arr.shape[0]} labels."
            )
        clean_trials = clean_trials[keep]

    return emb[keep], labels_arr[keep], clean_trials


def _majority_label(values: np.ndarray) -> Any:
    unique, counts = np.unique(values, return_counts=True)
    return unique[np.argmax(counts)]


def split_trials_for_alignment_and_decoding(
    labels: Any,
    trial_ids: Any,
    test_size: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sample indices into alignment and decoding/test sets by trial ID.

    Stratification is attempted using each trial's majority task label. If the
    trial label distribution is too sparse for stratification, the function
    falls back to an unstratified trial split.
    """
    labels_arr = np.asarray(labels).squeeze()
    trials = np.asarray(trial_ids).squeeze()
    if labels_arr.ndim != 1 or trials.ndim != 1 or labels_arr.shape[0] != trials.shape[0]:
        raise ValueError("labels and trial_ids must be one-dimensional arrays of matching length.")

    valid_trial_mask = _label_not_nan_mask(trials)
    if not np.all(valid_trial_mask):
        labels_arr = labels_arr[valid_trial_mask]
        trials = trials[valid_trial_mask]
        original_indices = np.flatnonzero(valid_trial_mask)
    else:
        original_indices = np.arange(len(labels_arr))

    unique_trials = np.unique(trials)
    if len(unique_trials) < 2:
        raise ValueError("At least two trials are required for trial-level alignment/decoding splits.")

    trial_labels = np.array([_majority_label(labels_arr[trials == trial]) for trial in unique_trials])
    stratify = trial_labels
    _, class_counts = np.unique(trial_labels, return_counts=True)
    if np.any(class_counts < 2) or len(unique_trials) * test_size < len(class_counts):
        stratify = None

    try:
        align_trials, decode_trials = train_test_split(
            unique_trials,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        align_trials, decode_trials = train_test_split(
            unique_trials,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    align_local = np.flatnonzero(np.isin(trials, align_trials))
    decode_local = np.flatnonzero(np.isin(trials, decode_trials))
    return original_indices[align_local], original_indices[decode_local]


def split_samples_for_alignment_and_decoding(
    labels: Any,
    test_size: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback split when trial IDs are unavailable.

    This avoids using the exact same samples for alignment and decoding, but it
    is weaker than trial-level splitting for time-series data. Output tables
    carry an explicit warning whenever this path is used.
    """
    labels_arr = np.asarray(labels).squeeze()
    indices = np.arange(len(labels_arr))
    stratify = labels_arr
    _, class_counts = np.unique(labels_arr, return_counts=True)
    if np.any(class_counts < 2):
        stratify = None
    try:
        align_idx, decode_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        align_idx, decode_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return np.asarray(align_idx), np.asarray(decode_idx)


def fit_zscore_transform(points: np.ndarray) -> ZScoreTransform:
    points = np.asarray(points, dtype=float)
    mean = np.mean(points, axis=0, keepdims=True)
    std = np.std(points, axis=0, ddof=0, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return ZScoreTransform(mean=mean, std=std)


def apply_zscore_transform(points: np.ndarray, transform: ZScoreTransform) -> np.ndarray:
    return (np.asarray(points, dtype=float) - transform.mean) / transform.std


def compute_task_bin_means(embeddings: Any, labels: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return one mean latent point for each sorted task label."""
    emb = _as_2d_embedding(embeddings)
    labels_arr = np.asarray(labels).squeeze()
    if len(labels_arr) != emb.shape[0]:
        raise ValueError("Embedding and label lengths do not match for task-bin means.")
    sorted_labels = np.array(sorted(np.unique(labels_arr)))
    means = []
    for label in sorted_labels:
        class_points = emb[labels_arr == label]
        if len(class_points) == 0:
            continue
        means.append(np.mean(class_points, axis=0))
    return np.asarray(means, dtype=float), sorted_labels


def _matched_task_bin_means(
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
    target_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_means, source_unique = compute_task_bin_means(source_embeddings, source_labels)
    target_means, target_unique = compute_task_bin_means(target_embeddings, target_labels)
    common = np.array([label for label in target_unique if label in set(source_unique)])
    if len(common) < 2:
        raise ValueError(f"At least two shared task labels are required for Procrustes alignment; got {common}.")

    source_lookup = {label: source_means[idx] for idx, label in enumerate(source_unique)}
    target_lookup = {label: target_means[idx] for idx, label in enumerate(target_unique)}
    source_points = np.vstack([source_lookup[label] for label in common])
    target_points = np.vstack([target_lookup[label] for label in common])
    return source_points, target_points, common


def fit_procrustes_transform(
    source_points: Any,
    target_points: Any,
    allow_scale: bool = False,
) -> ProcrustesTransform:
    """Fit an orthogonal transform mapping source points into target space.

    `source_points` are the test-rat alignment points and `target_points` are
    the train-rat alignment points. The transform centers both sets, estimates
    an orthogonal rotation with scipy.linalg.orthogonal_procrustes, and stores
    the centers so new source samples can be mapped as:

        aligned = (source - source_center) @ R + target_center

    Isotropic scaling is disabled by default so scale handling is explicit.
    """
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_points and target_points must both be 2D arrays.")
    if source.shape != target.shape:
        raise ValueError(f"source_points and target_points must have the same shape; got {source.shape} vs {target.shape}.")
    if source.shape[0] < 2:
        raise ValueError("At least two paired points are required for Procrustes alignment.")

    source_center = np.mean(source, axis=0, keepdims=True)
    target_center = np.mean(target, axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center
    rotation, scale_stat = orthogonal_procrustes(source_centered, target_centered)

    scale_factor = 1.0
    if allow_scale:
        rotated = source_centered @ rotation
        denom = np.sum(rotated**2)
        if denom > 0:
            scale_factor = float(np.sum(rotated * target_centered) / denom)

    return ProcrustesTransform(
        rotation=rotation,
        source_center=source_center,
        target_center=target_center,
        procrustes_scale_stat=float(scale_stat),
        allow_scale=allow_scale,
        scale_factor=scale_factor,
    )


def apply_procrustes_transform(points: Any, transform: ProcrustesTransform) -> np.ndarray:
    points_arr = np.asarray(points, dtype=float)
    aligned = (points_arr - transform.source_center) @ transform.rotation
    if transform.allow_scale:
        aligned = aligned * transform.scale_factor
    return aligned + transform.target_center


def _make_decoder(name: str, random_state: Optional[int] = None):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unknown decoder '{name}'. Choose one of: logreg, lda, knn.")


def _score_decoder(
    decoder_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels_order: np.ndarray,
    random_state: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    if len(np.unique(train_y)) < 2:
        raise ValueError("Decoder training labels contain fewer than two classes.")
    model = _make_decoder(decoder_name, random_state=random_state)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    acc = accuracy_score(test_y, pred)
    bal_acc = balanced_accuracy_score(test_y, pred)
    conf = confusion_matrix(test_y, pred, labels=labels_order)
    return float(acc), float(bal_acc), conf


def _json_confusion(conf: np.ndarray) -> str:
    return json.dumps(np.asarray(conf, dtype=int).tolist())


def _sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def cross_rat_latent_decode(
    train_emb: Any,
    train_labels: Any,
    test_emb: Any,
    test_labels: Any,
    train_trial_ids: Optional[Any] = None,
    test_trial_ids: Optional[Any] = None,
    decoder: str = "logreg",
    n_splits: int = 20,
    n_shuffles: int = 100,
    random_state: int = 0,
    train_rat: str = "train",
    test_rat: str = "test",
    task_scheme: str = "",
    dim: Optional[int] = None,
    model_run: Optional[int] = None,
    shuffle_test_labels: bool = False,
    test_size: float = 0.5,
    allow_scale: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Decode task epochs across rats after safe held-out latent alignment.

    Trial IDs trigger Option A: alignment transforms are estimated only from
    alignment trials, and decoder performance is evaluated on held-out decoding
    trials. If either rat lacks trial IDs, the function uses sample-level
    alignment/decoding splits and flags every output row with a warning.
    """
    train_x, train_y, train_trials = clean_labels_and_embeddings(train_emb, train_labels, train_trial_ids)
    test_x, test_y, test_trials = clean_labels_and_embeddings(test_emb, test_labels, test_trial_ids)
    if dim is None:
        dim = train_x.shape[1]
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError(f"Train/test latent dimensions differ: {train_x.shape[1]} vs {test_x.shape[1]}.")

    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    confusions: Dict[str, np.ndarray] = {}
    split_strategy = "trial" if train_trials is not None and test_trials is not None else "sample_fallback"
    warning_text = ""
    if split_strategy == "sample_fallback":
        warning_text = (
            "Trial IDs unavailable for at least one rat; alignment uses task-label means "
            "from sample-level alignment splits, and decoding is evaluated on held-out samples."
        )

    labels_order = np.array(sorted(np.unique(np.concatenate([train_y, test_y]))))
    chance_level = 1.0 / len(labels_order) if len(labels_order) else np.nan

    for split_idx in range(n_splits):
        split_seed_train = int(rng.integers(0, np.iinfo(np.int32).max))
        split_seed_test = int(rng.integers(0, np.iinfo(np.int32).max))

        if split_strategy == "trial":
            train_align_idx, train_decode_idx = split_trials_for_alignment_and_decoding(
                train_y, train_trials, test_size=test_size, random_state=split_seed_train
            )
            test_align_idx, test_decode_idx = split_trials_for_alignment_and_decoding(
                test_y, test_trials, test_size=test_size, random_state=split_seed_test
            )
        else:
            train_align_idx, train_decode_idx = split_samples_for_alignment_and_decoding(
                train_y, test_size=test_size, random_state=split_seed_train
            )
            test_align_idx, test_decode_idx = split_samples_for_alignment_and_decoding(
                test_y, test_size=test_size, random_state=split_seed_test
            )

        train_z = apply_zscore_transform(train_x, fit_zscore_transform(train_x[train_align_idx]))
        test_z = apply_zscore_transform(test_x, fit_zscore_transform(test_x[test_align_idx]))

        source_points, target_points, alignment_labels = _matched_task_bin_means(
            test_z[test_align_idx],
            test_y[test_align_idx],
            train_z[train_align_idx],
            train_y[train_align_idx],
        )
        transform = fit_procrustes_transform(source_points, target_points, allow_scale=allow_scale)
        aligned_test_decode = apply_procrustes_transform(test_z[test_decode_idx], transform)

        train_decode_x = train_z[train_decode_idx]
        train_decode_y = train_y[train_decode_idx]
        test_decode_y = test_y[test_decode_idx]

        alignment_warning = warning_text
        if len(alignment_labels) <= train_z.shape[1]:
            rank_note = (
                f"Only {len(alignment_labels)} task-label means were available to align "
                f"{train_z.shape[1]}D latent spaces; the rotation may be underdetermined."
            )
            alignment_warning = f"{alignment_warning} {rank_note}".strip()

        try:
            acc, bal_acc, conf = _score_decoder(
                decoder, train_decode_x, train_decode_y, aligned_test_decode, test_decode_y, labels_order, split_seed_train
            )
        except ValueError as exc:
            warnings.warn(f"Skipping real split {split_idx} for {train_rat}->{test_rat}: {exc}")
            continue

        key = f"{task_scheme}_dim{dim}_{train_rat}_to_{test_rat}_split{split_idx}_real"
        confusions[key] = conf
        rows.append(
            {
                "train_rat": train_rat,
                "test_rat": test_rat,
                "is_diagonal": train_rat == test_rat,
                "split_number": split_idx,
                "model_run": model_run,
                "shuffle_number": np.nan,
                "performance_type": "real",
                "task_scheme": task_scheme,
                "dim": dim,
                "decoder": decoder,
                "split_strategy": split_strategy,
                "n_train_samples": len(train_decode_idx),
                "n_test_samples": len(test_decode_idx),
                "n_alignment_train_samples": len(train_align_idx),
                "n_alignment_test_samples": len(test_align_idx),
                "n_alignment_labels": len(alignment_labels),
                "alignment_labels": json.dumps([str(x) for x in alignment_labels]),
                "chance_level": chance_level,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "confusion_matrix_key": key,
                "confusion_matrix": _json_confusion(conf),
                "alignment_warning": alignment_warning,
            }
        )

        for shuffle_idx in range(n_shuffles):
            shuffle_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            shuffle_rng = np.random.default_rng(shuffle_seed)
            shuffled_train_y = np.array(train_decode_y, copy=True)
            shuffle_rng.shuffle(shuffled_train_y)
            scoring_y = np.array(test_decode_y, copy=True)
            if shuffle_test_labels:
                shuffle_rng.shuffle(scoring_y)

            try:
                s_acc, s_bal_acc, s_conf = _score_decoder(
                    decoder,
                    train_decode_x,
                    shuffled_train_y,
                    aligned_test_decode,
                    scoring_y,
                    labels_order,
                    shuffle_seed,
                )
            except ValueError as exc:
                warnings.warn(f"Skipping shuffle {shuffle_idx} split {split_idx} for {train_rat}->{test_rat}: {exc}")
                continue

            s_key = f"{task_scheme}_dim{dim}_{train_rat}_to_{test_rat}_split{split_idx}_shuffle{shuffle_idx}"
            confusions[s_key] = s_conf
            rows.append(
                {
                    "train_rat": train_rat,
                    "test_rat": test_rat,
                    "is_diagonal": train_rat == test_rat,
                    "split_number": split_idx,
                    "model_run": model_run,
                    "shuffle_number": shuffle_idx,
                    "performance_type": "train_label_shuffle" if not shuffle_test_labels else "train_and_test_label_shuffle",
                    "task_scheme": task_scheme,
                    "dim": dim,
                    "decoder": decoder,
                    "split_strategy": split_strategy,
                    "n_train_samples": len(train_decode_idx),
                    "n_test_samples": len(test_decode_idx),
                    "n_alignment_train_samples": len(train_align_idx),
                    "n_alignment_test_samples": len(test_align_idx),
                    "n_alignment_labels": len(alignment_labels),
                    "alignment_labels": json.dumps([str(x) for x in alignment_labels]),
                    "chance_level": chance_level,
                    "accuracy": s_acc,
                    "balanced_accuracy": s_bal_acc,
                    "confusion_matrix_key": s_key,
                    "confusion_matrix": _json_confusion(s_conf),
                    "alignment_warning": alignment_warning,
                }
            )

    return pd.DataFrame(rows), confusions


def _find_nested_key(obj: Any, key_candidates: Sequence[str]) -> Optional[Any]:
    if isinstance(obj, Mapping):
        for key in key_candidates:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = _find_nested_key(value, key_candidates)
            if found is not None:
                return found
    if hasattr(obj, "_fieldnames"):
        for key in key_candidates:
            if key in obj._fieldnames:
                return getattr(obj, key)
        for field in obj._fieldnames:
            found = _find_nested_key(getattr(obj, field), key_candidates)
            if found is not None:
                return found
    return None


def _available_mat_keys(obj: Any, prefix: str = "", max_depth: int = 2) -> List[str]:
    if max_depth < 0:
        return []
    keys: List[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if key.startswith("__"):
                continue
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full)
            keys.extend(_available_mat_keys(value, full, max_depth - 1))
    elif hasattr(obj, "_fieldnames"):
        for field in obj._fieldnames:
            full = f"{prefix}.{field}" if prefix else str(field)
            keys.append(full)
            keys.extend(_available_mat_keys(getattr(obj, field), full, max_depth - 1))
    return keys


def _loadmat_flexible(path: str) -> Dict[str, Any]:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def _infer_rat_id(path: str, mat_data: Mapping[str, Any]) -> str:
    rat_value = _find_nested_key(mat_data, ("rat_id", "rat", "animal_id", "animal"))
    if rat_value is not None and np.asarray(rat_value).size == 1:
        return str(np.asarray(rat_value).item())
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(rat[_-]?[A-Za-z0-9]+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("-", "_")
    return stem


def load_cross_rat_data_from_mat(
    input_dir: str,
    task_schemes: Sequence[str],
    dims: Sequence[int],
    embedding_key: str = "embedding",
    label_key: str = "labels",
    trial_key: str = "trial_ids",
    file_pattern: str = "*.mat",
) -> Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
    """Load a flexible nested data dictionary from .mat files.

    The returned structure is:
        data[rat_id][task_scheme][dim] = {
            "embedding": samples x dim,
            "labels": samples,
            "trial_ids": optional samples,
        }
    """
    paths = sorted(glob(os.path.join(input_dir, file_pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {os.path.join(input_dir, file_pattern)}")

    data: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {}
    embedding_keys = tuple(dict.fromkeys((embedding_key, *EMBEDDING_KEY_CANDIDATES)))
    trial_keys = tuple(dict.fromkeys((trial_key, *TRIAL_KEY_CANDIDATES)))

    for path in paths:
        mat_data = _loadmat_flexible(path)
        rat_id = _infer_rat_id(path, mat_data)
        embedding = _find_nested_key(mat_data, embedding_keys)
        if embedding is None:
            print(f"WARNING: Missing embedding key in {path}. Available keys: {_available_mat_keys(mat_data)}")
            continue

        trial_ids = _find_nested_key(mat_data, trial_keys)
        file_lower = os.path.basename(path).lower()
        for task_scheme in task_schemes:
            task_specific_label_keys = (task_scheme, task_scheme.lower())
            labels = _find_nested_key(mat_data, task_specific_label_keys)
            if labels is None:
                generic_label_keys = tuple(dict.fromkeys((label_key, "labels", "label", "training", "y")))
                generic_labels = _find_nested_key(mat_data, generic_label_keys)
                if len(task_schemes) == 1 or task_scheme.lower() in file_lower or label_key.lower() == task_scheme.lower():
                    labels = generic_labels
            if labels is None:
                if task_scheme.lower() in file_lower:
                    print(f"WARNING: Missing labels for {task_scheme} in {path}. Available keys: {_available_mat_keys(mat_data)}")
                elif len(task_schemes) > 1:
                    print(
                        f"WARNING: Skipping {path} for {task_scheme}; no task-specific key was found. "
                        f"Use --label_key {task_scheme} or include {task_scheme} in the filename if this file is for that scheme."
                    )
                continue

            try:
                clean_embedding = _as_2d_embedding(embedding, np.asarray(labels).squeeze())
            except ValueError as exc:
                print(f"WARNING: Skipping {path} for {task_scheme}: {exc}")
                continue

            for dim in dims:
                if clean_embedding.shape[1] < dim:
                    continue
                entry = {
                    "embedding": clean_embedding[:, :dim],
                    "labels": np.asarray(labels).squeeze(),
                }
                if trial_ids is not None:
                    entry["trial_ids"] = np.asarray(trial_ids).squeeze()
                data.setdefault(rat_id, {}).setdefault(task_scheme, {})[dim] = entry

    if not data:
        raise ValueError("No usable rat/task/dim entries were loaded. Check keys and file_pattern.")
    return data


def _task_scheme_matches_npz(task_scheme: str, npz_data: Mapping[str, Any]) -> bool:
    if "task_scheme" in npz_data and str(np.asarray(npz_data["task_scheme"]).item()).lower() == task_scheme.lower():
        return True
    if "dimensions" in npz_data:
        try:
            n_bins = int(np.asarray(npz_data["dimensions"]).item())
            if task_scheme.upper() == f"CSUS{n_bins}":
                return True
        except Exception:
            pass
    if "bins" in npz_data:
        try:
            n_bins = len(np.asarray(npz_data["bins"]).squeeze())
            if task_scheme.upper() == f"CSUS{n_bins}":
                return True
        except Exception:
            pass
    return False


def load_cross_rat_data_from_npz(
    input_dir: str,
    task_schemes: Sequence[str],
    dims: Sequence[int],
    file_pattern: str = "*.npz",
    environment: str = "A",
) -> Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
    """Load enhanced geometry-preservation .npz files for decoding.

    Expected keys are written by `run_geometry_preservation`: `embeddingA_runs`
    or `embeddingB_runs`, matching `labelsA`/`labelsB`, and optional
    `trial_idsA`/`trial_idsB`. Existing older geometry files that only contain
    `zA_runs`/`zB_runs` are skipped because they contain task-bin means rather
    than per-sample embeddings.
    """
    env = environment.upper()
    if env not in {"A", "B"}:
        raise ValueError("--npz_environment must be A or B.")

    paths = sorted(glob(os.path.join(input_dir, file_pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {os.path.join(input_dir, file_pattern)}")

    embedding_key = f"embedding{env}_runs"
    label_key = f"labels{env}"
    trial_key = f"trial_ids{env}"
    data: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {}

    for path in paths:
        npz_data = np.load(path, allow_pickle=True)
        if embedding_key not in npz_data or label_key not in npz_data:
            print(
                f"WARNING: Skipping {path}; missing {embedding_key}/{label_key}. "
                "Re-run scripts/cond_geometry_preservation_script.py after the per-sample embedding-save update."
            )
            continue

        rat_id = str(np.asarray(npz_data["rat_id"]).item()) if "rat_id" in npz_data else _infer_rat_id(path, {})
        embedding_runs = np.asarray(npz_data[embedding_key], dtype=float)
        labels = np.asarray(npz_data[label_key]).squeeze()
        if embedding_runs.ndim == 2:
            embedding_runs = embedding_runs[np.newaxis, :, :]
        if embedding_runs.ndim != 3:
            print(f"WARNING: Skipping {path}; {embedding_key} must be runs x samples x dims, got {embedding_runs.shape}.")
            continue
        if embedding_runs.shape[1] != len(labels):
            print(
                f"WARNING: Skipping {path}; {embedding_key} sample count {embedding_runs.shape[1]} "
                f"does not match {label_key} length {len(labels)}."
            )
            continue

        trial_ids = None
        if trial_key in npz_data:
            candidate_trials = np.asarray(npz_data[trial_key]).squeeze()
            if candidate_trials.size == len(labels):
                trial_ids = candidate_trials

        for task_scheme in task_schemes:
            if not _task_scheme_matches_npz(task_scheme, npz_data):
                continue
            for dim in dims:
                if embedding_runs.shape[2] < dim:
                    continue
                entry = {
                    "embedding_runs": embedding_runs[:, :, :dim],
                    "labels": labels,
                }
                if trial_ids is not None:
                    entry["trial_ids"] = trial_ids
                data.setdefault(rat_id, {}).setdefault(task_scheme, {})[dim] = entry

    if not data:
        raise ValueError(
            "No usable enhanced .npz entries were loaded. If these are older geometry files, "
            "re-run cond_geometry_preservation_script.py so embeddingA_runs/embeddingB_runs are saved."
        )
    return data


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if results.empty:
        return pd.DataFrame(rows)

    group_cols = ["task_scheme", "dim", "train_rat", "test_rat", "is_diagonal"]
    real = results[results["performance_type"] == "real"]
    shuffle = results[results["performance_type"] != "real"]
    real_pair = real.groupby(group_cols, dropna=False).agg(
        real_accuracy_mean=("accuracy", "mean"),
        real_accuracy_sem=("accuracy", _sem),
        real_balanced_accuracy_mean=("balanced_accuracy", "mean"),
        real_balanced_accuracy_sem=("balanced_accuracy", _sem),
        n_real_observations=("accuracy", "size"),
        chance_level=("chance_level", "mean"),
    )
    shuffle_pair = shuffle.groupby(group_cols, dropna=False).agg(
        shuffle_accuracy_mean=("accuracy", "mean"),
        shuffle_accuracy_sem=("accuracy", _sem),
        shuffle_balanced_accuracy_mean=("balanced_accuracy", "mean"),
        shuffle_balanced_accuracy_sem=("balanced_accuracy", _sem),
        n_shuffle_observations=("accuracy", "size"),
    )
    pair_summary = real_pair.join(shuffle_pair, how="left").reset_index()
    pair_summary["accuracy_effect_real_minus_shuffle"] = (
        pair_summary["real_accuracy_mean"] - pair_summary["shuffle_accuracy_mean"]
    )
    pair_summary["balanced_accuracy_effect_real_minus_shuffle"] = (
        pair_summary["real_balanced_accuracy_mean"] - pair_summary["shuffle_balanced_accuracy_mean"]
    )
    pair_summary["row_type"] = "pair"
    rows.extend(pair_summary.to_dict("records"))

    for (task_scheme, dim), group in pair_summary.groupby(["task_scheme", "dim"], dropna=False):
        offdiag = group[~group["is_diagonal"].astype(bool)]
        if offdiag.empty:
            continue
        real_values = offdiag["real_accuracy_mean"].to_numpy(dtype=float)
        shuffle_values = offdiag["shuffle_accuracy_mean"].to_numpy(dtype=float)
        real_balanced_values = offdiag["real_balanced_accuracy_mean"].to_numpy(dtype=float)
        shuffle_balanced_values = offdiag["shuffle_balanced_accuracy_mean"].to_numpy(dtype=float)
        valid = np.isfinite(real_values) & np.isfinite(shuffle_values)
        real_values = real_values[valid]
        shuffle_values = shuffle_values[valid]
        diffs = real_values - shuffle_values
        balanced_valid = np.isfinite(real_balanced_values) & np.isfinite(shuffle_balanced_values)
        real_balanced_values = real_balanced_values[balanced_valid]
        shuffle_balanced_values = shuffle_balanced_values[balanced_valid]
        balanced_diffs = real_balanced_values - shuffle_balanced_values
        t_p = np.nan
        wilcoxon_p = np.nan
        if len(diffs) >= 2 and np.any(np.abs(diffs) > 0):
            try:
                t_p = float(ttest_rel(real_values, shuffle_values, nan_policy="omit").pvalue)
            except Exception:
                t_p = np.nan
            try:
                wilcoxon_p = float(wilcoxon(real_values, shuffle_values).pvalue)
            except Exception:
                wilcoxon_p = np.nan
        balanced_t_p = np.nan
        balanced_wilcoxon_p = np.nan
        if len(balanced_diffs) >= 2 and np.any(np.abs(balanced_diffs) > 0):
            try:
                balanced_t_p = float(ttest_rel(real_balanced_values, shuffle_balanced_values, nan_policy="omit").pvalue)
            except Exception:
                balanced_t_p = np.nan
            try:
                balanced_wilcoxon_p = float(wilcoxon(real_balanced_values, shuffle_balanced_values).pvalue)
            except Exception:
                balanced_wilcoxon_p = np.nan

        rows.append(
            {
                "row_type": "aggregate_off_diagonal",
                "task_scheme": task_scheme,
                "dim": dim,
                "train_rat": "OFF_DIAGONAL_MEAN",
                "test_rat": "OFF_DIAGONAL_MEAN",
                "is_diagonal": False,
                "real_accuracy_mean": float(np.mean(real_values)) if len(real_values) else np.nan,
                "real_accuracy_sem": _sem(real_values),
                "shuffle_accuracy_mean": float(np.mean(shuffle_values)) if len(shuffle_values) else np.nan,
                "shuffle_accuracy_sem": _sem(shuffle_values),
                "accuracy_effect_real_minus_shuffle": float(np.mean(diffs)) if len(diffs) else np.nan,
                "accuracy_effect_sem": _sem(diffs),
                "paired_t_p_value": t_p,
                "wilcoxon_p_value": wilcoxon_p,
                "real_balanced_accuracy_mean": float(np.mean(real_balanced_values)) if len(real_balanced_values) else np.nan,
                "real_balanced_accuracy_sem": _sem(real_balanced_values),
                "shuffle_balanced_accuracy_mean": float(np.mean(shuffle_balanced_values)) if len(shuffle_balanced_values) else np.nan,
                "shuffle_balanced_accuracy_sem": _sem(shuffle_balanced_values),
                "balanced_accuracy_effect_real_minus_shuffle": float(np.mean(balanced_diffs)) if len(balanced_diffs) else np.nan,
                "balanced_accuracy_effect_sem": _sem(balanced_diffs),
                "balanced_accuracy_paired_t_p_value": balanced_t_p,
                "balanced_accuracy_wilcoxon_p_value": balanced_wilcoxon_p,
                "n_pairs": int(len(diffs)),
                "chance_level": float(offdiag["chance_level"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _matrix_from_summary(
    summary: pd.DataFrame,
    rats: Sequence[str],
    task_scheme: str,
    dim: int,
    column: str,
) -> np.ndarray:
    matrix = np.full((len(rats), len(rats)), np.nan, dtype=float)
    pair_rows = summary[
        (summary["row_type"] == "pair") & (summary["task_scheme"] == task_scheme) & (summary["dim"] == dim)
    ]
    rat_to_idx = {rat: idx for idx, rat in enumerate(rats)}
    for _, row in pair_rows.iterrows():
        if row["train_rat"] in rat_to_idx and row["test_rat"] in rat_to_idx:
            matrix[rat_to_idx[row["train_rat"]], rat_to_idx[row["test_rat"]]] = row[column]
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    rats: Sequence[str],
    title: str,
    output_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    diagonal_outline: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5.0, 0.6 * len(rats) + 2.0), max(4.5, 0.55 * len(rats) + 1.8)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f3f4f6")
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(len(rats)))
    ax.set_yticks(np.arange(len(rats)))
    ax.set_xticklabels(rats, rotation=45, ha="right")
    ax.set_yticklabels(rats)
    ax.set_xlabel("Test rat")
    ax.set_ylabel("Train rat")
    ax.set_title(title)
    for i in range(len(rats)):
        for j in range(len(rats)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
        if diagonal_outline:
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=1.4))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(
    summary: pd.DataFrame,
    task_scheme: str,
    dim: int,
    output_path: str,
) -> None:
    pair_rows = summary[
        (summary["row_type"] == "pair") & (summary["task_scheme"] == task_scheme) & (summary["dim"] == dim)
    ]
    if pair_rows.empty:
        return
    offdiag = pair_rows[~pair_rows["is_diagonal"].astype(bool)]
    diag = pair_rows[pair_rows["is_diagonal"].astype(bool)]
    bars = []
    if not offdiag.empty:
        bars.append(("Across-rat real", offdiag["real_accuracy_mean"].mean(), _sem(offdiag["real_accuracy_mean"])))
        bars.append(("Across-rat shuffle", offdiag["shuffle_accuracy_mean"].mean(), _sem(offdiag["shuffle_accuracy_mean"])))
    if not diag.empty:
        bars.append(("Within-rat control", diag["real_accuracy_mean"].mean(), _sem(diag["real_accuracy_mean"])))
    if not bars:
        return

    labels, means, errors = zip(*bars)
    colors = ["#2563eb", "#9ca3af", "#059669"][: len(labels)]
    fig, ax = plt.subplots(figsize=(max(5.2, 1.5 * len(labels)), 4.2))
    ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=4, color=colors, edgecolor="#111827", linewidth=0.6)
    chance = float(pair_rows["chance_level"].mean())
    if np.isfinite(chance):
        ax.axhline(chance, color="#dc2626", linestyle="--", linewidth=1.1, label=f"Chance ({chance:.2f})")
        ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"{task_scheme} dim {dim} decoding summary")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_mat_compatible_output(results: pd.DataFrame, summary: pd.DataFrame, output_path: str) -> None:
    mat_dict: Dict[str, Any] = {}
    for name, frame in (("results", results), ("summary", summary)):
        for column in frame.columns:
            values = frame[column].to_numpy()
            key = f"{name}_{column}"
            if np.issubdtype(values.dtype, np.number) or values.dtype == bool:
                mat_dict[key] = values
            else:
                mat_dict[key] = values.astype(str)
    savemat(output_path, mat_dict, do_compression=True)


def run_all_cross_rat_decoding(
    data: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    output_dir: str,
    task_schemes: Sequence[str],
    dims: Sequence[int],
    decoder: str = "logreg",
    n_splits: int = 20,
    n_shuffles: int = 100,
    random_state: int = 0,
    shuffle_test_labels: bool = False,
    allow_scale: bool = False,
    save_mat: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)
    all_results: List[pd.DataFrame] = []
    all_confusions: Dict[str, np.ndarray] = {}
    rats = sorted(data.keys())

    for task_scheme in task_schemes:
        for dim in dims:
            available_rats = [
                rat
                for rat in rats
                if task_scheme in data.get(rat, {}) and dim in data[rat].get(task_scheme, {})
            ]
            if len(available_rats) < 2:
                print(f"WARNING: Skipping {task_scheme} dim {dim}; fewer than two rats have usable entries.")
                continue

            for train_rat in available_rats:
                for test_rat in available_rats:
                    train_entry = data[train_rat][task_scheme][dim]
                    test_entry = data[test_rat][task_scheme][dim]
                    train_runs = train_entry.get("embedding_runs")
                    test_runs = test_entry.get("embedding_runs")
                    if train_runs is None:
                        train_runs = np.asarray(train_entry["embedding"])[np.newaxis, :, :]
                    if test_runs is None:
                        test_runs = np.asarray(test_entry["embedding"])[np.newaxis, :, :]
                    n_model_runs = min(len(train_runs), len(test_runs))
                    for model_run in range(n_model_runs):
                        pair_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                        print(
                            f"Running {task_scheme} dim {dim}: train {train_rat} -> test {test_rat}, "
                            f"model_run {model_run}"
                        )
                        pair_results, pair_confusions = cross_rat_latent_decode(
                            train_runs[model_run],
                            train_entry["labels"],
                            test_runs[model_run],
                            test_entry["labels"],
                            train_trial_ids=train_entry.get("trial_ids"),
                            test_trial_ids=test_entry.get("trial_ids"),
                            decoder=decoder,
                            n_splits=n_splits,
                            n_shuffles=n_shuffles,
                            random_state=pair_seed,
                            train_rat=train_rat,
                            test_rat=test_rat,
                            task_scheme=task_scheme,
                            dim=dim,
                            model_run=model_run,
                            shuffle_test_labels=shuffle_test_labels,
                            allow_scale=allow_scale,
                        )
                        all_results.append(pair_results)
                        all_confusions.update(pair_confusions)

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary = summarize_results(results)

    results_path = os.path.join(output_dir, "cross_rat_decoding_results.csv")
    summary_path = os.path.join(output_dir, "cross_rat_decoding_summary.csv")
    confusion_path = os.path.join(output_dir, "confusion_matrices.npz")
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    np.savez_compressed(confusion_path, **all_confusions)

    if save_mat:
        save_mat_compatible_output(results, summary, os.path.join(output_dir, "cross_rat_decoding_results.mat"))

    for task_scheme in task_schemes:
        for dim in dims:
            pair_rows = summary[
                (summary.get("row_type", pd.Series(dtype=str)) == "pair")
                & (summary.get("task_scheme", pd.Series(dtype=str)) == task_scheme)
                & (summary.get("dim", pd.Series(dtype=int)) == dim)
            ]
            if pair_rows.empty:
                continue
            plot_rats = sorted(set(pair_rows["train_rat"]).union(pair_rows["test_rat"]))
            real_matrix = _matrix_from_summary(summary, plot_rats, task_scheme, dim, "real_accuracy_mean")
            shuffle_matrix = _matrix_from_summary(summary, plot_rats, task_scheme, dim, "shuffle_accuracy_mean")
            finite_values = np.concatenate([real_matrix[np.isfinite(real_matrix)], shuffle_matrix[np.isfinite(shuffle_matrix)]])
            vmin, vmax = (0.0, 1.0) if finite_values.size == 0 else (max(0.0, float(np.min(finite_values))), min(1.0, float(np.max(finite_values))))
            if math.isclose(vmin, vmax):
                vmin, vmax = 0.0, 1.0
            plot_heatmap(
                real_matrix,
                plot_rats,
                f"Real cross-rat decoding accuracy: {task_scheme} dim {dim}",
                os.path.join(output_dir, f"heatmap_real_accuracy_{task_scheme}_dim{dim}.svg"),
                vmin=vmin,
                vmax=vmax,
            )
            plot_heatmap(
                shuffle_matrix,
                plot_rats,
                f"Shuffle cross-rat decoding accuracy: {task_scheme} dim {dim}",
                os.path.join(output_dir, f"heatmap_shuffle_accuracy_{task_scheme}_dim{dim}.svg"),
                vmin=vmin,
                vmax=vmax,
            )
            plot_summary_bar(
                summary,
                task_scheme,
                dim,
                os.path.join(output_dir, f"summary_bar_{task_scheme}_dim{dim}.svg"),
            )

    return results, summary


def parse_csv_list(values: str, cast=str) -> List[Any]:
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Across-animal latent-space task-epoch decoding for CEBRA embeddings. "
            "The decoder is trained on one rat's latent coordinates and tested on "
            "another rat after held-out Procrustes alignment."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .mat files or enhanced geometry .npz files.")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV, NPZ, MAT, and SVG outputs.")
    parser.add_argument("--input_format", choices=["auto", "mat", "npz"], default="auto", help="Input file format.")
    parser.add_argument("--task_schemes", default="CSUS2,CSUS5", help="Comma-separated task label schemes.")
    parser.add_argument("--dims", default="2,3,5,7,10", help="Comma-separated latent dimensions to test.")
    parser.add_argument("--n_splits", type=int, default=20, help="Number of alignment/decoding splits per ordered rat pair.")
    parser.add_argument("--n_shuffles", type=int, default=100, help="Number of shuffle controls per split.")
    parser.add_argument("--decoder", choices=["logreg", "lda", "knn"], default="logreg", help="Decoder type.")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed.")
    parser.add_argument("--embedding_key", default="embedding", help="Preferred .mat key for latent embeddings.")
    parser.add_argument("--label_key", default="labels", help="Preferred .mat key for task labels.")
    parser.add_argument("--trial_key", default="trial_ids", help="Preferred .mat key for trial IDs.")
    parser.add_argument("--file_pattern", default=None, help="Glob pattern inside input_dir. Defaults to *.mat or *.npz.")
    parser.add_argument("--npz_environment", choices=["A", "B"], default="A", help="For enhanced geometry .npz files, use A or B per-sample embeddings.")
    parser.add_argument("--shuffle_test_labels", action="store_true", help="Also shuffle test labels before scoring shuffle controls.")
    parser.add_argument("--allow_scale", action="store_true", help="Allow isotropic scaling after orthogonal Procrustes rotation.")
    parser.add_argument("--no_mat", action="store_true", help="Skip MAT-compatible output.")
    args = parser.parse_args()

    task_schemes = parse_csv_list(args.task_schemes, str)
    dims = parse_csv_list(args.dims, int)
    input_format = args.input_format
    if input_format == "auto":
        has_npz = bool(glob(os.path.join(args.input_dir, args.file_pattern or "*.npz")))
        has_mat = bool(glob(os.path.join(args.input_dir, args.file_pattern or "*.mat")))
        input_format = "npz" if has_npz else "mat" if has_mat else "mat"

    if input_format == "npz":
        data = load_cross_rat_data_from_npz(
            args.input_dir,
            task_schemes=task_schemes,
            dims=dims,
            file_pattern=args.file_pattern or "*.npz",
            environment=args.npz_environment,
        )
    else:
        data = load_cross_rat_data_from_mat(
            args.input_dir,
            task_schemes=task_schemes,
            dims=dims,
            embedding_key=args.embedding_key,
            label_key=args.label_key,
            trial_key=args.trial_key,
            file_pattern=args.file_pattern or "*.mat",
        )
    results, summary = run_all_cross_rat_decoding(
        data,
        output_dir=args.output_dir,
        task_schemes=task_schemes,
        dims=dims,
        decoder=args.decoder,
        n_splits=args.n_splits,
        n_shuffles=args.n_shuffles,
        random_state=args.random_state,
        shuffle_test_labels=args.shuffle_test_labels,
        allow_scale=args.allow_scale,
        save_mat=not args.no_mat,
    )

    print(f"Saved results to {os.path.join(args.output_dir, 'cross_rat_decoding_results.csv')}")
    print(f"Saved summary to {os.path.join(args.output_dir, 'cross_rat_decoding_summary.csv')}")
    print(f"Saved confusion matrices to {os.path.join(args.output_dir, 'confusion_matrices.npz')}")
    if not results.empty and results["split_strategy"].eq("sample_fallback").any():
        print(
            "WARNING: Some analyses used sample-level fallback splits because trial IDs were unavailable. "
            "See alignment_warning in the results CSV."
        )
    aggregate = summary[summary.get("row_type", pd.Series(dtype=str)) == "aggregate_off_diagonal"]
    if not aggregate.empty:
        print("Off-diagonal aggregate accuracy:")
        for _, row in aggregate.iterrows():
            print(
                f"  {row['task_scheme']} dim {int(row['dim'])}: "
                f"real={row['real_accuracy_mean']:.4f} +/- {row['real_accuracy_sem']:.4f}, "
                f"shuffle={row['shuffle_accuracy_mean']:.4f} +/- {row['shuffle_accuracy_sem']:.4f}, "
                f"diff={row['accuracy_effect_real_minus_shuffle']:.4f}, "
                f"paired t p={row.get('paired_t_p_value', np.nan):.4g}, "
                f"Wilcoxon p={row.get('wilcoxon_p_value', np.nan):.4g}"
            )


if __name__ == "__main__":
    main()
