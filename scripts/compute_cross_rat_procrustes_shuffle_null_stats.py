import argparse
import glob
import os
import re
import sys
from datetime import datetime
from itertools import combinations, permutations

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import t


COMPARISONS = [
    ("A_vs_A", "A_vs_A", "zA_runs", "zA_runs"),
    ("A_vs_B", "A_vs_B", "zA_runs", "zB_runs"),
    ("B_vs_A", "A_vs_B", "zB_runs", "zA_runs"),
    ("B_vs_B", "B_vs_B", "zB_runs", "zB_runs"),
]
FULL_POP_COMPARISON_MODE = "An_vs_B1_separately_trained_full_population"
GEOMETRY_NPZ_RE = re.compile(
    r"^geometry_preservation_(rat[^_]+)_.+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.npz$"
)


def load_scalar_string(data, key):
    if key not in data:
        return ""
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 0:
        return ""
    return str(value.reshape(-1)[0])


def load_rat_id(data, npz_path):
    if "rat_id" in data:
        rat_id = str(data["rat_id"])
        if rat_id:
            return rat_id
    return os.path.basename(npz_path).split("_")[2] if "_" in os.path.basename(npz_path) else os.path.basename(npz_path)


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


def procrustes_metrics(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
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


def load_rat_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(f"{npz_path} does not contain zA_runs/zB_runs.")

    z_a_runs = np.asarray(data["zA_runs"], dtype=float)
    z_b_runs = np.asarray(data["zB_runs"], dtype=float)
    if z_a_runs.shape != z_b_runs.shape:
        raise ValueError(f"{npz_path} has mismatched zA_runs/zB_runs shapes: {z_a_runs.shape} vs {z_b_runs.shape}.")

    return {
        "path": npz_path,
        "rat_id": load_rat_id(data, npz_path),
        "zA_runs": z_a_runs,
        "zB_runs": z_b_runs,
    }


def build_mean_ab_runs(rat, zscore_before_within_align=True):
    z_a_runs = rat["zA_runs"]
    z_b_runs = rat["zB_runs"]
    mean_ab_runs = np.zeros_like(z_a_runs, dtype=float)

    for run_idx in range(z_a_runs.shape[0]):
        z_a = z_a_runs[run_idx]
        z_b = z_b_runs[run_idx]
        if zscore_before_within_align:
            z_a = zscore_columns(z_a)
            z_b = zscore_columns(z_b)

        z_b_aligned = procrustes_align(z_b, z_a)
        mean_ab_runs[run_idx] = (z_a + z_b_aligned) / 2.0

    return mean_ab_runs


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(len(values))


def summarize_metric(actual_pair_means, null_means, metric_name):
    actual_pair_means = np.asarray(actual_pair_means, dtype=float)
    null_means = np.asarray(null_means, dtype=float)
    actual_mean = np.mean(actual_pair_means)
    actual_sem = sem(actual_pair_means)
    if len(actual_pair_means) > 1:
        ci_half_width = t.ppf(0.975, df=len(actual_pair_means) - 1) * actual_sem
    else:
        ci_half_width = np.nan

    # Lower Procrustes disparity/residual is better, so the one-sided p-value
    # counts null means as or more aligned than the observed mean.
    p_lower = (np.sum(null_means <= actual_mean) + 1) / (len(null_means) + 1)

    return {
        f"actual_mean_{metric_name}": actual_mean,
        f"actual_sem_{metric_name}": actual_sem,
        f"actual_ci95_low_{metric_name}": actual_mean - ci_half_width,
        f"actual_ci95_high_{metric_name}": actual_mean + ci_half_width,
        f"shuffle_null_mean_{metric_name}": np.mean(null_means),
        f"shuffle_null_5th_percentile_{metric_name}": np.percentile(null_means, 5),
        f"shuffle_null_95th_percentile_{metric_name}": np.percentile(null_means, 95),
        f"one_sided_p_lower_than_shuffle_{metric_name}": p_lower,
        f"actual_minus_shuffle_null_mean_{metric_name}": actual_mean - np.mean(null_means),
    }


def build_pair_run_observations(rat_npz_paths, zscore_before_align=True):
    rats = [load_rat_embeddings(path) for path in rat_npz_paths]

    observations = []
    for rat_left, rat_right in combinations(rats, 2):
        for comparison, comparison_family, left_key, right_key in COMPARISONS:
            left_runs = rat_left[left_key]
            right_runs = rat_right[right_key]
            n_runs = min(left_runs.shape[0], right_runs.shape[0])
            if left_runs.shape[1:] != right_runs.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for {rat_left['rat_id']}__{rat_right['rat_id']} {comparison}: "
                    f"{left_runs.shape[1:]} vs {right_runs.shape[1:]}"
                )

            for run_idx in range(n_runs):
                target_embedding = left_runs[run_idx]
                source_embedding = right_runs[run_idx]
                if zscore_before_align:
                    target_embedding = zscore_columns(target_embedding)
                    source_embedding = zscore_columns(source_embedding)
                observations.append(
                    {
                        "rat_left": rat_left["rat_id"],
                        "rat_right": rat_right["rat_id"],
                        "rat_pair": f"{rat_left['rat_id']}__{rat_right['rat_id']}",
                        "comparison": comparison,
                        "comparison_family": comparison_family,
                        "model_run": run_idx,
                        "target_embedding": target_embedding,
                        "source_embedding": source_embedding,
                    }
                )

    return observations


def compute_observation_tables(observations):
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]
    rows = []
    for obs_idx, obs in enumerate(observations):
        metric_values = procrustes_metrics(obs["source_embedding"], obs["target_embedding"])
        row = {
            "observation_id": obs_idx,
            "comparison": obs["comparison"],
            "comparison_family": obs["comparison_family"],
            "rat_left": obs["rat_left"],
            "rat_right": obs["rat_right"],
            "rat_pair": obs["rat_pair"],
            "model_run": obs["model_run"],
        }
        for metric in metrics:
            row[metric] = metric_values[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_shuffle_metric_matrices(observations):
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]
    n_bins = observations[0]["source_embedding"].shape[0]
    permutations_by_bin = nonidentity_permutations(n_bins)
    shuffle_by_metric = {
        metric: np.empty((len(observations), len(permutations_by_bin)), dtype=float)
        for metric in metrics
    }

    for obs_idx, obs in enumerate(observations):
        source = obs["source_embedding"]
        target = obs["target_embedding"]
        for perm_idx, permutation in enumerate(permutations_by_bin):
            metric_values = procrustes_metrics(source[permutation], target)
            for metric in metrics:
                shuffle_by_metric[metric][obs_idx, perm_idx] = metric_values[metric]

    return shuffle_by_metric, permutations_by_bin


def pair_mean_table(real_scores):
    return (
        real_scores.groupby(["comparison_family", "rat_pair"], dropna=False)
        .agg(
            actual_pair_mean_procrustes_disparity=("procrustes_disparity", "mean"),
            actual_pair_mean_aligned_rmse=("aligned_rmse", "mean"),
            actual_pair_mean_aligned_sse_over_target_ss=("aligned_sse_over_target_ss", "mean"),
            n_model_runs=("model_run", "size"),
        )
        .reset_index()
    )


def compute_null_means(real_scores, shuffle_by_metric, permutations_by_bin, n_null, rng):
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]
    null_rows = []

    for comparison_family, family_rows in real_scores.groupby("comparison_family", dropna=False):
        observation_ids = family_rows["observation_id"].to_numpy(dtype=int)
        rat_pairs = family_rows["rat_pair"].to_numpy()
        unique_rat_pairs = np.unique(rat_pairs)
        draw_idx = rng.integers(0, len(permutations_by_bin), size=(n_null, len(observation_ids)))
        for null_idx in range(n_null):
            row = {
                "comparison_family": comparison_family,
                "null_sample": null_idx,
                "n_rat_pairs": len(unique_rat_pairs),
            }
            for metric in metrics:
                shuffled_scores = shuffle_by_metric[metric][observation_ids, draw_idx[null_idx]]
                pair_null_means = []
                for rat_pair in unique_rat_pairs:
                    pair_null_means.append(np.mean(shuffled_scores[rat_pairs == rat_pair]))
                row[f"shuffle_null_mean_{metric}"] = np.mean(pair_null_means)
            null_rows.append(row)

    return pd.DataFrame(null_rows)


def summarize_with_null(pair_means, null_means):
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]
    rows = []
    for comparison_family, family_pair_means in pair_means.groupby("comparison_family", dropna=False):
        family_null_means = null_means[null_means["comparison_family"] == comparison_family]
        summary = {
            "comparison_family": comparison_family,
            "n_rat_pairs": len(family_pair_means),
            "n_null_mean_samples": len(family_null_means),
        }
        for metric in metrics:
            actual_pair_means = family_pair_means[f"actual_pair_mean_{metric}"].to_numpy(dtype=float)
            null_values = family_null_means[f"shuffle_null_mean_{metric}"].to_numpy(dtype=float)
            summary.update(summarize_metric(actual_pair_means, null_values, metric))
        rows.append(summary)
    return pd.DataFrame(rows)


def build_plot_data(pair_means, null_means):
    metrics = ["procrustes_disparity", "aligned_rmse", "aligned_sse_over_target_ss"]
    rows = []
    for _, row in pair_means.iterrows():
        out = {
            "comparison_family": row["comparison_family"],
            "score_type": "actual_rat_pair_mean",
            "rat_pair": row["rat_pair"],
            "null_sample": np.nan,
        }
        for metric in metrics:
            out[metric] = row[f"actual_pair_mean_{metric}"]
        rows.append(out)

    for _, row in null_means.iterrows():
        out = {
            "comparison_family": row["comparison_family"],
            "score_type": "shuffle_null_mean",
            "rat_pair": "",
            "null_sample": row["null_sample"],
        }
        for metric in metrics:
            out[metric] = row[f"shuffle_null_mean_{metric}"]
        rows.append(out)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute cross-rat Procrustes residual/disparity statistics with "
            "the same 20-run/500-null task-bin shuffle paradigm used for the "
            "within-rat Procrustes analysis. Cross-rat Procrustes metrics are "
            "computed for all rat pairs and comparison families A_vs_A, A_vs_B "
            "(including both A_vs_B and B_vs_A directions), and B_vs_B. Each "
            "null sample shuffles the source rat's task-bin order within every "
            "rat-pair/run comparison, averages within rat pairs, then averages "
            "across rat pairs. Lower Procrustes metrics indicate better alignment."
        )
    )
    parser.add_argument("rat_npz", nargs="*", help="Per-rat geometry_preservation .npz files containing zA_runs/zB_runs.")
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory containing geometry_preservation_rat*.npz files. When provided, "
            "the newest non-checkpoint NPZ is selected for each rat."
        ),
    )
    parser.add_argument("--n_null", type=int, default=500, help="Number of cross-rat mean null samples.")
    parser.add_argument("--random_seed", type=int, default=20260508)
    parser.add_argument("--output_dir", default="cross_rat_procrustes_shuffle_null_outputs")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before cross-rat alignment.")
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
    required_comparison_mode = None if args.require_comparison_mode == "any" else args.require_comparison_mode
    if bool(args.input_dir) == bool(args.rat_npz):
        raise ValueError("Provide either --input_dir or explicit rat_npz paths, but not both.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    if required_comparison_mode:
        for path in rat_npz_paths:
            with np.load(path, allow_pickle=True) as data:
                comparison_mode = load_scalar_string(data, "comparison_mode")
            if comparison_mode != required_comparison_mode:
                raise ValueError(
                    f"{path} has comparison_mode={comparison_mode or '<missing>'}; "
                    f"expected {required_comparison_mode}. Use --require_comparison_mode any to override."
                )

    observations = build_pair_run_observations(rat_npz_paths, zscore_before_align=not args.no_zscore)
    real_scores = compute_observation_tables(observations)
    shuffle_by_metric, permutations_by_bin = compute_shuffle_metric_matrices(observations)
    pair_means = pair_mean_table(real_scores)
    null_means = compute_null_means(real_scores, shuffle_by_metric, permutations_by_bin, args.n_null, rng)
    stats = summarize_with_null(pair_means, null_means)
    plot_data = build_plot_data(pair_means, null_means)

    run_scores_path = os.path.join(args.output_dir, f"cross_rat_procrustes_run_scores_{timestamp}.csv")
    pair_means_path = os.path.join(args.output_dir, f"cross_rat_procrustes_actual_rat_pair_means_{timestamp}.csv")
    null_means_path = os.path.join(args.output_dir, f"cross_rat_procrustes_shuffle_null_means_{timestamp}.csv")
    stats_path = os.path.join(args.output_dir, f"cross_rat_procrustes_shuffle_null_stats_{timestamp}.csv")
    plot_data_path = os.path.join(args.output_dir, f"cross_rat_procrustes_shuffle_null_plot_data_{timestamp}.csv")

    real_scores.to_csv(run_scores_path, index=False)
    pair_means.to_csv(pair_means_path, index=False)
    null_means.to_csv(null_means_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_data.to_csv(plot_data_path, index=False)

    print(f"Saved run-level actual scores to {run_scores_path}")
    print(f"Saved actual rat-pair means to {pair_means_path}")
    print(f"Saved shuffle null means to {null_means_path}")
    print(f"Saved cross-rat Procrustes shuffle-null stats to {stats_path}")
    print(f"Saved long-form plotting data to {plot_data_path}")
    print(
        stats[
            [
                "comparison_family",
                "n_rat_pairs",
                "actual_mean_procrustes_disparity",
                "actual_sem_procrustes_disparity",
                "actual_ci95_low_procrustes_disparity",
                "actual_ci95_high_procrustes_disparity",
                "shuffle_null_mean_procrustes_disparity",
                "shuffle_null_5th_percentile_procrustes_disparity",
                "one_sided_p_lower_than_shuffle_procrustes_disparity",
                "actual_mean_aligned_rmse",
                "shuffle_null_mean_aligned_rmse",
                "one_sided_p_lower_than_shuffle_aligned_rmse",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
