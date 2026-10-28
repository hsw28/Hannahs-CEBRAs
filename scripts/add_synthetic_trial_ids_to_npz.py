#!/usr/bin/env python3
"""Copy geometry-preservation NPZ files and add inferred trial IDs.

The geometry-preservation outputs used for cross-rat latent decoding may have
empty trial_id arrays even when samples remain in repeated trial order. This
script infers the shortest repeated label block for each label vector and
writes trial IDs that keep each repeated block together.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Dict, Tuple

import numpy as np


TRIAL_KEYS = {
    "labelsA": "trial_idsA",
    "labelsB": "trial_idsB",
    "labelsAn": "trial_idsAn",
}


def infer_repeated_block(labels: np.ndarray) -> Tuple[int, int]:
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got {labels.shape}.")
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("Cannot infer trial IDs from an empty label vector.")

    for period in range(1, n_samples + 1):
        if n_samples % period != 0:
            continue
        block = labels[:period]
        tiled = np.tile(block, n_samples // period)
        if np.array_equal(labels, tiled):
            if len(np.unique(block)) < 2:
                continue
            return period, n_samples // period

    raise ValueError("Could not infer a repeated multi-label trial block.")


def make_trial_ids(labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
    period, n_trials = infer_repeated_block(labels)
    return np.repeat(np.arange(n_trials, dtype=np.int64), period), period, n_trials


def patch_npz(input_path: str, output_path: str) -> Dict[str, Tuple[int, int]]:
    data = np.load(input_path, allow_pickle=True)
    copied = {key: data[key] for key in data.files}
    inferred: Dict[str, Tuple[int, int]] = {}

    for label_key, trial_key in TRIAL_KEYS.items():
        if label_key not in copied:
            continue
        trial_ids, period, n_trials = make_trial_ids(copied[label_key])
        if len(trial_ids) != len(np.asarray(copied[label_key]).squeeze()):
            raise ValueError(f"Inferred trial IDs for {label_key} have the wrong length.")
        copied[trial_key] = trial_ids
        inferred[trial_key] = (period, n_trials)

    np.savez(output_path, **copied)
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory containing geometry-preservation .npz files.")
    parser.add_argument("output_dir", help="Directory for patched .npz copies.")
    parser.add_argument("--pattern", default="*.npz", help="Input glob pattern.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

    for input_path in paths:
        output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        inferred = patch_npz(input_path, output_path)
        summary = ", ".join(
            f"{key}: block={period}, trials={n_trials}"
            for key, (period, n_trials) in sorted(inferred.items())
        )
        print(f"Wrote {output_path}")
        print(f"  {summary}")


if __name__ == "__main__":
    main()
