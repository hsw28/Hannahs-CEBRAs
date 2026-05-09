import argparse
import os
import sys
from itertools import combinations, permutations

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import numpy as np
import pandas as pd
from scipy.stats import t

from cond_geometry_preservation import geometry_preservation_score


COMPARISONS = [
    ("A_vs_A", "A_vs_A", "zA_runs", "zA_runs"),
    ("A_vs_B", "A_vs_B", "zA_runs", "zB_runs"),
    ("B_vs_A", "A_vs_B", "zB_runs", "zA_runs"),
    ("B_vs_B", "B_vs_B", "zB_runs", "zB_runs"),
]


def load_rat_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(f"{npz_path} does not contain zA_runs/zB_runs.")

    rat_id = str(data["rat_id"]) if "rat_id" in data else ""
    if not rat_id:
        rat_id = os.path.basename(npz_path).split("_")[2] if "_" in os.path.basename(npz_path) else os.path.basename(npz_path)

    return {
        "path": npz_path,
        "rat_id": rat_id,
        "zA_runs": np.asarray(data["zA_runs"], dtype=float),
        "zB_runs": np.asarray(data["zB_runs"], dtype=float),
    }


def nonidentity_permutations(n_bins):
    identity = tuple(range(n_bins))
    return np.array([perm for perm in permutations(range(n_bins)) if perm != identity], dtype=int)


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(len(values))


def build_cross_rat_observations(rat_npz_paths):
    rats = [load_rat_embeddings(path) for path in rat_npz_paths]
    observations = []

    for rat_left, rat_right in combinations(rats, 2):
        rat_pair = f"{rat_left['rat_id']}__{rat_right['rat_id']}"
        for comparison, comparison_family, left_key, right_key in COMPARISONS:
            left_runs = rat_left[left_key]
            right_runs = rat_right[right_key]
            n_runs = min(left_runs.shape[0], right_runs.shape[0])

            if left_runs.shape[1:] != right_runs.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for {rat_pair} {comparison}: "
                    f"{left_runs.shape[1:]} vs {right_runs.shape[1:]}"
                )

            for run_idx in range(n_runs):
                observations.append(
                    {
                        "rat_left": rat_left["rat_id"],
                        "rat_right": rat_right["rat_id"],
                        "rat_pair": rat_pair,
                        "comparison": comparison,
                        "comparison_family": comparison_family,
                        "model_run": run_idx,
                        "left_embedding": left_runs[run_idx],
                        "right_embedding": right_runs[run_idx],
                    }
                )

    return observations


def compute_observation_scores(observations):
    rows = []
    for obs_idx, obs in enumerate(observations):
        rows.append(
            {
                "observation_id": obs_idx,
                "rat_left": obs["rat_left"],
                "rat_right": obs["rat_right"],
                "rat_pair": obs["rat_pair"],
                "comparison": obs["comparison"],
                "comparison_family": obs["comparison_family"],
                "model_run": obs["model_run"],
                "rReal": geometry_preservation_score(obs["left_embedding"], obs["right_embedding"]),
            }
        )
    return pd.DataFrame(rows)


def compute_shuffle_score_matrix(observations):
    if not observations:
        raise ValueError("No observations were generated.")

    n_bins = observations[0]["right_embedding"].shape[0]
    permutations_by_bin = nonidentity_permutations(n_bins)
    shuffle_scores = np.empty((len(observations), len(permutations_by_bin)), dtype=float)

    for obs_idx, obs in enumerate(observations):
        left_embedding = obs["left_embedding"]
        right_embedding = obs["right_embedding"]
        if right_embedding.shape[0] != n_bins:
            raise ValueError("All observations must have the same number of task bins.")
        for perm_idx, permutation in enumerate(permutations_by_bin):
            shuffle_scores[obs_idx, perm_idx] = geometry_preservation_score(
                left_embedding,
                right_embedding[permutation],
            )

    return shuffle_scores, permutations_by_bin


def pair_mean_table(real_scores):
    return (
        real_scores.groupby(["comparison_family", "rat_pair"], dropna=False)
        .agg(
            actual_pair_mean=("rReal", "mean"),
            n_scores=("rReal", "size"),
        )
        .reset_index()
    )


def family_actual_stats(pair_means):
    rows = []
    for comparison_family, group in pair_means.groupby("comparison_family", dropna=False):
        actual_values = group["actual_pair_mean"].to_numpy(dtype=float)
        actual_mean = np.mean(actual_values)
        actual_sem = sem(actual_values)
        if len(actual_values) > 1:
            ci_half_width = t.ppf(0.975, df=len(actual_values) - 1) * actual_sem
        else:
            ci_half_width = np.nan
        rows.append(
            {
                "comparison_family": comparison_family,
                "n_rat_pairs": len(actual_values),
                "actual_mean": actual_mean,
                "actual_sem": actual_sem,
                "actual_ci95_low": actual_mean - ci_half_width,
                "actual_ci95_high": actual_mean + ci_half_width,
            }
        )
    return pd.DataFrame(rows)


def compute_null_means(real_scores, shuffle_scores, permutations_by_bin, n_null, rng):
    null_rows = []
    observation_ids = real_scores["observation_id"].to_numpy(dtype=int)

    for comparison_family, family_rows in real_scores.groupby("comparison_family", dropna=False):
        family_observation_ids = family_rows["observation_id"].to_numpy(dtype=int)
        family_rat_pairs = family_rows["rat_pair"].to_numpy()
        unique_rat_pairs = np.unique(family_rat_pairs)

        for null_idx in range(n_null):
            perm_idx = rng.integers(0, len(permutations_by_bin), size=len(family_observation_ids))
            shuffled_scores = shuffle_scores[family_observation_ids, perm_idx]

            pair_null_means = []
            for rat_pair in unique_rat_pairs:
                pair_null_means.append(np.mean(shuffled_scores[family_rat_pairs == rat_pair]))
            null_rows.append(
                {
                    "comparison_family": comparison_family,
                    "null_sample": null_idx,
                    "shuffle_null_mean": np.mean(pair_null_means),
                    "n_rat_pairs": len(pair_null_means),
                }
            )

    return pd.DataFrame(null_rows)


def summarize_with_null(pair_means, null_means):
    actual = family_actual_stats(pair_means)
    rows = []

    for _, actual_row in actual.iterrows():
        comparison_family = actual_row["comparison_family"]
        null_values = null_means.loc[
            null_means["comparison_family"] == comparison_family,
            "shuffle_null_mean",
        ].to_numpy(dtype=float)
        actual_mean = actual_row["actual_mean"]
        p_one_sided = (np.sum(null_values >= actual_mean) + 1) / (len(null_values) + 1)

        row = actual_row.to_dict()
        row.update(
            {
                "n_null_mean_samples": len(null_values),
                "shuffle_null_mean": np.mean(null_values),
                "shuffle_null_95th_percentile": np.percentile(null_values, 95),
                "one_sided_permutation_p": p_one_sided,
                "actual_minus_shuffle_null_mean": actual_mean - np.mean(null_values),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_plot_data(pair_means, null_means):
    rows = []
    for _, row in pair_means.iterrows():
        rows.append(
            {
                "comparison_family": row["comparison_family"],
                "score_type": "actual_rat_pair_mean",
                "score": row["actual_pair_mean"],
                "rat_pair": row["rat_pair"],
                "null_sample": np.nan,
            }
        )

    for _, row in null_means.iterrows():
        rows.append(
            {
                "comparison_family": row["comparison_family"],
                "score_type": "shuffle_null_mean",
                "score": row["shuffle_null_mean"],
                "rat_pair": "",
                "null_sample": row["null_sample"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute cross-rat matrix/RSA geometry-preservation statistics with "
            "a 500-sample task-bin correspondence shuffle null. The observed "
            "statistic for each comparison family is the mean across rat-pair "
            "means. Each null sample shuffles the right-side task-bin order "
            "within every cross-rat model-run comparison, averages shuffled "
            "scores within each rat pair, then averages across rat pairs."
        )
    )
    parser.add_argument("rat_npz", nargs="+", help="Per-rat geometry_preservation .npz files containing zA_runs/zB_runs.")
    parser.add_argument("--n_null", type=int, default=500, help="Number of cross-rat mean null samples per comparison family.")
    parser.add_argument("--random_seed", type=int, default=20260508)
    parser.add_argument("--output_dir", default="cross_rat_geometry_shuffle_null_outputs")
    args = parser.parse_args()

    if args.n_null < 1:
        raise ValueError("--n_null must be at least 1.")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    observations = build_cross_rat_observations(args.rat_npz)
    real_scores = compute_observation_scores(observations)
    shuffle_scores, permutations_by_bin = compute_shuffle_score_matrix(observations)
    pair_means = pair_mean_table(real_scores)
    null_means = compute_null_means(real_scores, shuffle_scores, permutations_by_bin, args.n_null, rng)
    stats = summarize_with_null(pair_means, null_means)
    plot_data = build_plot_data(pair_means, null_means)

    real_scores_path = os.path.join(args.output_dir, "cross_rat_geometry_run_scores.csv")
    pair_means_path = os.path.join(args.output_dir, "cross_rat_geometry_actual_rat_pair_means.csv")
    null_means_path = os.path.join(args.output_dir, "cross_rat_geometry_shuffle_null_means.csv")
    stats_path = os.path.join(args.output_dir, "cross_rat_geometry_shuffle_null_stats.csv")
    plot_data_path = os.path.join(args.output_dir, "cross_rat_geometry_shuffle_null_plot_data.csv")

    real_scores.to_csv(real_scores_path, index=False)
    pair_means.to_csv(pair_means_path, index=False)
    null_means.to_csv(null_means_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_data.to_csv(plot_data_path, index=False)

    print(f"Saved run-level actual scores to {real_scores_path}")
    print(f"Saved actual rat-pair means to {pair_means_path}")
    print(f"Saved shuffle null means to {null_means_path}")
    print(f"Saved cross-rat shuffle-null stats to {stats_path}")
    print(f"Saved long-form plotting data to {plot_data_path}")
    print(
        stats[
            [
                "comparison_family",
                "n_rat_pairs",
                "actual_mean",
                "actual_sem",
                "actual_ci95_low",
                "actual_ci95_high",
                "shuffle_null_mean",
                "shuffle_null_95th_percentile",
                "one_sided_permutation_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
