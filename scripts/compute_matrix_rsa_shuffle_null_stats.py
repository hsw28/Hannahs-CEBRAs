import argparse
import glob
import os
import re
import sys
from datetime import datetime
from itertools import permutations

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, t


FULL_POP_COMPARISON_MODE = "An_vs_B1_separately_trained_full_population"
GEOMETRY_NPZ_RE = re.compile(
    r"^geometry_preservation_(rat[^_]+)_.+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.npz$"
)


def zscore_columns(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def geometry_preservation_score(z_a, z_b):
    z_a = zscore_columns(z_a)
    z_b = zscore_columns(z_b)
    return spearmanr(pdist(z_a, metric="euclidean"), pdist(z_b, metric="euclidean")).correlation


def nonidentity_permutations(n_bins):
    identity = tuple(range(n_bins))
    return np.array([perm for perm in permutations(range(n_bins)) if perm != identity], dtype=int)


def load_rat_id(data, npz_path):
    if "rat_id" in data:
        rat_id = str(data["rat_id"])
        if rat_id:
            return rat_id
    return os.path.basename(npz_path).split("_")[2]


def load_scalar_string(data, key):
    if key not in data:
        return ""
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 0:
        return ""
    return str(value.reshape(-1)[0])


def geometry_npz_sort_key(npz_path):
    basename = os.path.basename(npz_path)
    match = GEOMETRY_NPZ_RE.match(basename)
    if match:
        return match.group(2)
    return f"mtime_{os.path.getmtime(npz_path):.6f}"


def discover_latest_rat_npzs(input_dir, required_comparison_mode):
    candidates = sorted(glob.glob(os.path.join(input_dir, "geometry_preservation_rat*.npz")))
    by_rat = {}
    skipped = []

    for path in candidates:
        basename = os.path.basename(path)
        if basename.endswith("_checkpoint.npz"):
            skipped.append((path, "checkpoint"))
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                if "zA_runs" not in data or "zB_runs" not in data:
                    skipped.append((path, "missing zA_runs/zB_runs"))
                    continue
                comparison_mode = load_scalar_string(data, "comparison_mode")
                if required_comparison_mode and comparison_mode != required_comparison_mode:
                    skipped.append((path, f"comparison_mode={comparison_mode or '<missing>'}"))
                    continue
                rat_id = load_rat_id(data, path)
        except Exception as exc:
            skipped.append((path, f"could not read: {exc}"))
            continue

        current = by_rat.get(rat_id)
        if current is None or geometry_npz_sort_key(path) > geometry_npz_sort_key(current):
            by_rat[rat_id] = path

    if not by_rat:
        details = "\n".join(f"  skipped {path}: {reason}" for path, reason in skipped[:20])
        raise ValueError(
            f"No usable final geometry NPZ files found in {input_dir}."
            + (f"\n{details}" if details else "")
        )

    return [by_rat[rat_id] for rat_id in sorted(by_rat)], skipped


def summarize_rat(npz_path, n_null, rng):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(f"{npz_path} does not contain zA_runs/zB_runs.")

    rat_id = load_rat_id(data, npz_path)
    z_a_runs = np.asarray(data["zA_runs"], dtype=float)
    z_b_runs = np.asarray(data["zB_runs"], dtype=float)
    if z_a_runs.shape != z_b_runs.shape:
        raise ValueError(f"{npz_path} has mismatched zA_runs/zB_runs shapes: {z_a_runs.shape} vs {z_b_runs.shape}.")

    n_runs, n_bins = z_a_runs.shape[:2]
    permutations_by_bin = nonidentity_permutations(n_bins)

    real_scores = np.array(
        [geometry_preservation_score(z_a_runs[run_idx], z_b_runs[run_idx]) for run_idx in range(n_runs)]
    )

    per_run_shuffle_scores = np.empty((n_runs, len(permutations_by_bin)), dtype=float)
    for run_idx in range(n_runs):
        for perm_idx, permutation in enumerate(permutations_by_bin):
            per_run_shuffle_scores[run_idx, perm_idx] = geometry_preservation_score(
                z_a_runs[run_idx],
                z_b_runs[run_idx][permutation],
            )

    draw_idx = rng.integers(0, len(permutations_by_bin), size=(n_null, n_runs))
    null_means = per_run_shuffle_scores[np.arange(n_runs), draw_idx].mean(axis=1)

    mean_real = np.mean(real_scores)
    sem_real = np.std(real_scores, ddof=1) / np.sqrt(n_runs)
    ci_half_width = t.ppf(0.975, df=n_runs - 1) * sem_real if n_runs > 1 else np.nan
    p_one_sided = (np.sum(null_means >= mean_real) + 1) / (len(null_means) + 1)

    summary = {
        "rat_id": rat_id,
        "n_model_runs": n_runs,
        "n_bins": n_bins,
        "n_nonidentity_bin_permutations_per_run": len(permutations_by_bin),
        "n_null_mean_samples": n_null,
        "mean_rReal": mean_real,
        "sem_rReal": sem_real,
        "ci95_low_rReal": mean_real - ci_half_width,
        "ci95_high_rReal": mean_real + ci_half_width,
        "shuffle_null_mean": np.mean(null_means),
        "shuffle_null_95th_percentile": np.percentile(null_means, 95),
        "one_sided_permutation_p": p_one_sided,
        "mean_rReal_minus_shuffle_null_mean": mean_real - np.mean(null_means),
        "source_npz": npz_path,
    }

    plot_rows = []
    for run_idx, score in enumerate(real_scores):
        plot_rows.append(
            {
                "rat": rat_id,
                "score_type": "real_run",
                "score": score,
                "model_run": run_idx,
                "null_sample": np.nan,
                "source_npz": npz_path,
            }
        )
    for null_idx, score in enumerate(null_means):
        plot_rows.append(
            {
                "rat": rat_id,
                "score_type": "shuffle_null_mean",
                "score": score,
                "model_run": np.nan,
                "null_sample": null_idx,
                "source_npz": npz_path,
            }
        )

    return summary, plot_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report per-rat matrix/RSA geometry-preservation statistics using a "
            "task-bin correspondence shuffle null across model runs."
        )
    )
    parser.add_argument("rat_npz", nargs="*", help="geometry_preservation .npz files containing zA_runs/zB_runs.")
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory containing geometry_preservation_rat*.npz files. When provided, "
            "the newest non-checkpoint NPZ is selected for each rat."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output summary CSV. Defaults to a timestamped file in geometry_matrix_stats_outputs/.",
    )
    parser.add_argument(
        "--plot_data_csv",
        default=None,
        help=(
            "Long-form CSV for plotting. Contains rat, score_type, score, model_run, "
            "and null_sample columns; this is the input expected by "
            "plot_geometry_permutation_null_violins.m. Defaults to a timestamped "
            "file in geometry_matrix_stats_outputs/."
        ),
    )
    parser.add_argument("--n_null", type=int, default=500, help="Number of mean-over-runs null samples per rat.")
    parser.add_argument("--random_seed", type=int, default=20260508)
    parser.add_argument(
        "--require_comparison_mode",
        default="any",
        help=(
            "Optional required comparison_mode in each NPZ. Defaults to 'any'. "
            f"Use '{FULL_POP_COMPARISON_MODE}' to require the full-population A(n)-vs-B(1) mode."
        ),
    )
    args = parser.parse_args()

    if args.n_null < 1:
        raise ValueError("--n_null must be at least 1.")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = args.output_csv or os.path.join(
        "geometry_matrix_stats_outputs",
        f"per_rat_matrix_rsa_shuffle_null_stats_{timestamp}.csv",
    )
    plot_data_csv = args.plot_data_csv or os.path.join(
        "geometry_matrix_stats_outputs",
        f"per_rat_matrix_rsa_shuffle_null_plot_data_{timestamp}.csv",
    )
    required_comparison_mode = None if args.require_comparison_mode == "any" else args.require_comparison_mode
    if bool(args.input_dir) == bool(args.rat_npz):
        raise ValueError("Provide either --input_dir or explicit rat_npz paths, but not both.")

    rng = np.random.default_rng(args.random_seed)
    if args.input_dir:
        rat_npz_paths, skipped = discover_latest_rat_npzs(args.input_dir, required_comparison_mode)
        print("Selected latest geometry NPZ per rat:")
        for path in rat_npz_paths:
            print(f"  {path}")
        if skipped:
            print(f"Skipped {len(skipped)} non-selected/unusable NPZ file(s); checkpoints and mismatched modes are ignored.")
    else:
        rat_npz_paths = args.rat_npz

    summary_rows = []
    plot_rows = []
    for path in rat_npz_paths:
        if required_comparison_mode:
            with np.load(path, allow_pickle=True) as data:
                comparison_mode = load_scalar_string(data, "comparison_mode")
            if comparison_mode != required_comparison_mode:
                raise ValueError(
                    f"{path} has comparison_mode={comparison_mode or '<missing>'}; "
                    f"expected {required_comparison_mode}. Use --require_comparison_mode any to override."
                )
        summary, rat_plot_rows = summarize_rat(path, args.n_null, rng)
        summary_rows.append(summary)
        plot_rows.extend(rat_plot_rows)
    results = pd.DataFrame(summary_rows).sort_values("rat_id")
    plot_data = pd.DataFrame(plot_rows).sort_values(["rat", "score_type", "null_sample", "model_run"])

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results.to_csv(output_csv, index=False)
    plot_output_dir = os.path.dirname(plot_data_csv)
    if plot_output_dir:
        os.makedirs(plot_output_dir, exist_ok=True)
    plot_data.to_csv(plot_data_csv, index=False)

    print(f"Saved per-rat matrix/RSA shuffle-null stats to {output_csv}")
    print(f"Saved long-form plotting data to {plot_data_csv}")
    print(
        results[
            [
                "rat_id",
                "n_model_runs",
                "mean_rReal",
                "sem_rReal",
                "ci95_low_rReal",
                "ci95_high_rReal",
                "shuffle_null_mean",
                "shuffle_null_95th_percentile",
                "one_sided_permutation_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
