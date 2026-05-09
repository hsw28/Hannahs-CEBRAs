import argparse
import os
import sys
from itertools import permutations

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import t


def zscore_columns(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def procrustes_align(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = np.mean(source, axis=0, keepdims=True)
    target_center = np.mean(target, axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center

    rotation, _ = orthogonal_procrustes(source_centered, target_centered)
    aligned = source_centered @ rotation
    denom = np.sum(aligned ** 2)
    if denom > 0:
        aligned = aligned * (np.sum(aligned * target_centered) / denom)
    return aligned + target_center


def procrustes_metrics(source, target, zscore=True):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if zscore:
        source = zscore_columns(source)
        target = zscore_columns(target)

    aligned = procrustes_align(source, target)
    _, _, disparity = procrustes(target, source)
    residual = aligned - target
    aligned_sse = np.sum(residual ** 2)
    aligned_rmse = np.sqrt(np.mean(residual ** 2))
    target_centered = target - np.mean(target, axis=0, keepdims=True)
    target_total_ss = np.sum(target_centered ** 2)
    if target_total_ss > 0:
        aligned_sse_over_target_ss = aligned_sse / target_total_ss
    else:
        aligned_sse_over_target_ss = np.nan

    return {
        "procrustes_disparity": float(disparity),
        "aligned_sse": float(aligned_sse),
        "aligned_rmse": float(aligned_rmse),
        "target_total_ss": float(target_total_ss),
        "aligned_sse_over_target_ss": float(aligned_sse_over_target_ss),
    }


def nonidentity_permutations(n_bins):
    identity = tuple(range(n_bins))
    return np.array([perm for perm in permutations(range(n_bins)) if perm != identity], dtype=int)


def load_rat_id(data, npz_path):
    if "rat_id" in data:
        rat_id = str(data["rat_id"])
        if rat_id:
            return rat_id
    return os.path.basename(npz_path).split("_")[2]


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(len(values))


def summarize_metric(real_scores, null_means, metric_name):
    real_scores = np.asarray(real_scores, dtype=float)
    null_means = np.asarray(null_means, dtype=float)

    observed_mean = np.mean(real_scores)
    observed_sem = sem(real_scores)
    if len(real_scores) > 1:
        ci_half_width = t.ppf(0.975, df=len(real_scores) - 1) * observed_sem
    else:
        ci_half_width = np.nan

    # Lower Procrustes disparity/residual is better, so the one-sided p-value
    # counts null means as or more aligned than the observed mean.
    p_lower = (np.sum(null_means <= observed_mean) + 1) / (len(null_means) + 1)

    return {
        f"mean_real_{metric_name}": observed_mean,
        f"sem_real_{metric_name}": observed_sem,
        f"ci95_low_real_{metric_name}": observed_mean - ci_half_width,
        f"ci95_high_real_{metric_name}": observed_mean + ci_half_width,
        f"shuffle_null_mean_{metric_name}": np.mean(null_means),
        f"shuffle_null_5th_percentile_{metric_name}": np.percentile(null_means, 5),
        f"shuffle_null_95th_percentile_{metric_name}": np.percentile(null_means, 95),
        f"one_sided_p_lower_than_shuffle_{metric_name}": p_lower,
        f"real_minus_shuffle_null_mean_{metric_name}": observed_mean - np.mean(null_means),
    }


def summarize_rat(npz_path, n_null, rng, zscore=True):
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
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]

    real_by_metric = {metric: np.zeros(n_runs, dtype=float) for metric in metrics}
    shuffle_by_metric = {
        metric: np.empty((n_runs, len(permutations_by_bin)), dtype=float)
        for metric in metrics
    }

    for run_idx in range(n_runs):
        real_metrics = procrustes_metrics(z_b_runs[run_idx], z_a_runs[run_idx], zscore=zscore)
        for metric in metrics:
            real_by_metric[metric][run_idx] = real_metrics[metric]

        for perm_idx, permutation in enumerate(permutations_by_bin):
            shuffled_metrics = procrustes_metrics(
                z_b_runs[run_idx][permutation],
                z_a_runs[run_idx],
                zscore=zscore,
            )
            for metric in metrics:
                shuffle_by_metric[metric][run_idx, perm_idx] = shuffled_metrics[metric]

    draw_idx = rng.integers(0, len(permutations_by_bin), size=(n_null, n_runs))
    summary = {
        "rat_id": rat_id,
        "n_model_runs": n_runs,
        "n_bins": n_bins,
        "n_nonidentity_bin_permutations_per_run": len(permutations_by_bin),
        "n_null_mean_samples": n_null,
        "zscore_before_align": zscore,
        "source_npz": npz_path,
    }

    null_means_by_metric = {}
    for metric in metrics:
        null_means = shuffle_by_metric[metric][np.arange(n_runs), draw_idx].mean(axis=1)
        null_means_by_metric[metric] = null_means
        summary.update(summarize_metric(real_by_metric[metric], null_means, metric))

    run_rows = []
    for run_idx in range(n_runs):
        row = {
            "rat": rat_id,
            "score_type": "real_run",
            "model_run": run_idx,
            "null_sample": np.nan,
            "source_npz": npz_path,
        }
        for metric in metrics:
            row[metric] = real_by_metric[metric][run_idx]
        run_rows.append(row)

    for null_idx in range(n_null):
        row = {
            "rat": rat_id,
            "score_type": "shuffle_null_mean",
            "model_run": np.nan,
            "null_sample": null_idx,
            "source_npz": npz_path,
        }
        for metric in metrics:
            row[metric] = null_means_by_metric[metric][null_idx]
        run_rows.append(row)

    return summary, run_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute within-rat Procrustes residual/disparity statistics with "
            "the same 20-run/500-null task-bin shuffle paradigm used for the "
            "matrix/RSA geometry-preservation analysis. For each rat, the "
            "observed statistic is the mean real B-to-A Procrustes metric across "
            "model runs. Each null sample permutes B task-bin correspondence "
            "within each model run, computes one shuffled Procrustes metric per "
            "run, and averages across runs. Lower Procrustes metrics indicate "
            "better alignment."
        )
    )
    parser.add_argument("rat_npz", nargs="+", help="Per-rat geometry_preservation .npz files containing zA_runs/zB_runs.")
    parser.add_argument("--n_null", type=int, default=500, help="Number of mean-over-runs null samples per rat.")
    parser.add_argument("--random_seed", type=int, default=20260508)
    parser.add_argument("--output_dir", default="procrustes_shuffle_null_outputs")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before alignment.")
    args = parser.parse_args()

    if args.n_null < 1:
        raise ValueError("--n_null must be at least 1.")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    zscore = not args.no_zscore

    summary_rows = []
    plot_rows = []
    for path in args.rat_npz:
        summary, rat_plot_rows = summarize_rat(path, args.n_null, rng, zscore=zscore)
        summary_rows.append(summary)
        plot_rows.extend(rat_plot_rows)

    stats = pd.DataFrame(summary_rows).sort_values("rat_id")
    plot_data = pd.DataFrame(plot_rows).sort_values(["rat", "score_type", "null_sample", "model_run"])

    stats_path = os.path.join(args.output_dir, "within_rat_procrustes_shuffle_null_stats.csv")
    plot_data_path = os.path.join(args.output_dir, "within_rat_procrustes_shuffle_null_plot_data.csv")
    stats.to_csv(stats_path, index=False)
    plot_data.to_csv(plot_data_path, index=False)

    print(f"Saved within-rat Procrustes shuffle-null stats to {stats_path}")
    print(f"Saved long-form plotting data to {plot_data_path}")
    print(
        stats[
            [
                "rat_id",
                "n_model_runs",
                "mean_real_procrustes_disparity",
                "sem_real_procrustes_disparity",
                "ci95_low_real_procrustes_disparity",
                "ci95_high_real_procrustes_disparity",
                "shuffle_null_mean_procrustes_disparity",
                "shuffle_null_5th_percentile_procrustes_disparity",
                "one_sided_p_lower_than_shuffle_procrustes_disparity",
                "mean_real_aligned_rmse",
                "shuffle_null_mean_aligned_rmse",
                "one_sided_p_lower_than_shuffle_aligned_rmse",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
