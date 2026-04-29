import argparse
import os
import sys
from datetime import datetime
from itertools import combinations

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import numpy as np
import pandas as pd

from cond_geometry_preservation import compute_geometry_preservation_run, paired_geometry_stats


def load_rat_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(
            f"{npz_path} does not contain zA_runs/zB_runs. "
            "Re-run cond_geometry_preservation_script.py after the embedding-save update."
        )

    rat_id = str(data["rat_id"]) if "rat_id" in data else ""
    if not rat_id:
        rat_id = os.path.basename(npz_path).split("_")[2] if "_" in os.path.basename(npz_path) else os.path.basename(npz_path)

    return {
        "path": npz_path,
        "rat_id": rat_id,
        "zA_runs": np.asarray(data["zA_runs"], dtype=float),
        "zB_runs": np.asarray(data["zB_runs"], dtype=float),
        "bins": np.asarray(data["bins"]) if "bins" in data else None,
    }


def compare_cross_rat_geometries(rat_embedding_files, n_shuff=1, random_seed=None):
    rng = np.random.default_rng(random_seed)
    rats = [load_rat_embeddings(path) for path in rat_embedding_files]
    rows = []
    shuffle_rows = []

    comparisons = [
        ("A_vs_A", "zA_runs", "zA_runs"),
        ("A_vs_B", "zA_runs", "zB_runs"),
        ("B_vs_A", "zB_runs", "zA_runs"),
        ("B_vs_B", "zB_runs", "zB_runs"),
    ]

    for rat_left, rat_right in combinations(rats, 2):
        for comparison, left_key, right_key in comparisons:
            left_runs = rat_left[left_key]
            right_runs = rat_right[right_key]
            n_runs = min(left_runs.shape[0], right_runs.shape[0])
            if left_runs.shape[1:] != right_runs.shape[1:]:
                raise ValueError(
                    f"Embedding shapes do not match for {rat_left['rat_id']} {comparison} {rat_right['rat_id']}: "
                    f"{left_runs.shape[1:]} vs {right_runs.shape[1:]}"
                )

            for run_idx in range(n_runs):
                result = compute_geometry_preservation_run(left_runs[run_idx], right_runs[run_idx], n_shuff=n_shuff, rng=rng)
                family = "A_vs_B" if comparison in ("A_vs_B", "B_vs_A") else comparison
                rows.append(
                    {
                        "rat_left": rat_left["rat_id"],
                        "rat_right": rat_right["rat_id"],
                        "rat_pair": f"{rat_left['rat_id']}__{rat_right['rat_id']}",
                        "comparison": comparison,
                        "comparison_family": family,
                        "model_run": run_idx,
                        "n_shuff": n_shuff,
                        "rReal": result["rReal"],
                        "rShuff": result["rShuff"],
                        "rDiff": result["rReal"] - result["rShuff"],
                    }
                )
                for shuffle_idx, score in enumerate(result["rShuffAll"]):
                    shuffle_rows.append(
                        {
                            "rat_left": rat_left["rat_id"],
                            "rat_right": rat_right["rat_id"],
                            "rat_pair": f"{rat_left['rat_id']}__{rat_right['rat_id']}",
                            "comparison": comparison,
                            "comparison_family": family,
                            "model_run": run_idx,
                            "shuffle_id": shuffle_idx,
                            "shuffle_score": score,
                        }
                    )

    results = pd.DataFrame(rows)
    shuffles = pd.DataFrame(shuffle_rows)
    stats_rows = []
    for comparison, group in results.groupby("comparison_family"):
        stats = paired_geometry_stats(group["rReal"], group["rShuff"], rng=rng)
        stats_rows.append({"comparison_family": comparison, "stats_unit": "run", **stats})

        pair_means = (
            group.groupby("rat_pair", dropna=False)
            .agg(rReal=("rReal", "mean"), rShuff=("rShuff", "mean"))
            .reset_index()
        )
        pair_stats = paired_geometry_stats(pair_means["rReal"], pair_means["rShuff"], rng=rng)
        stats_rows.append({"comparison_family": comparison, "stats_unit": "rat_pair_mean", **pair_stats})
    stats = pd.DataFrame(stats_rows)
    return results, shuffles, stats


def plot_cross_rat_results(results, stats, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["A_vs_A", "A_vs_B", "B_vs_B"]
    colors = {"A_vs_A": "#2563eb", "A_vs_B": "#7c3aed", "B_vs_B": "#c2410c"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x_positions = np.arange(len(order))
    for x, comparison in zip(x_positions, order):
        group = (
            results[results["comparison_family"] == comparison]
            .groupby("rat_pair", dropna=False)
            .agg(rReal=("rReal", "mean"), rShuff=("rShuff", "mean"))
            .reset_index()
        )
        if group.empty:
            continue
        jitter = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(group), x) + jitter,
            group["rReal"],
            s=24,
            alpha=0.55,
            color=colors[comparison],
            edgecolor="none",
        )
        stat_row = stats[(stats["comparison_family"] == comparison) & (stats["stats_unit"] == "rat_pair_mean")].iloc[0]
        ax.errorbar(
            x,
            stat_row["real_mean"],
            yerr=stat_row["real_sem"],
            color="#111827",
            marker="_",
            markersize=24,
            capsize=4,
            linewidth=1.4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["A v A", "A v B", "B v B"])
    ax.set_ylabel("Cross-rat Spearman geometry score")
    ax.set_title("Cross-Rat Task Geometry")
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CEBRA task-distance geometry across rats using per-rat "
            "geometry_preservation .npz files containing zA_runs/zB_runs."
        )
    )
    parser.add_argument("rat_npz", nargs="+", help="Per-rat geometry_preservation *.npz files.")
    parser.add_argument("--n_shuff", type=int, default=1, help="Number of shuffled controls per cross-rat run comparison.")
    parser.add_argument("--output_dir", default="cross_rat_geometry_outputs", help="Directory for CSV and plot outputs.")
    parser.add_argument("--random_seed", type=int, default=None, help="Optional random seed for shuffle controls.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results, shuffles, stats = compare_cross_rat_geometries(args.rat_npz, n_shuff=args.n_shuff, random_seed=args.random_seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(args.output_dir, f"cross_rat_geometry_results_{timestamp}.csv")
    shuffles_path = os.path.join(args.output_dir, f"cross_rat_geometry_shuffles_{timestamp}.csv")
    stats_path = os.path.join(args.output_dir, f"cross_rat_geometry_stats_{timestamp}.csv")
    plot_path = os.path.join(args.output_dir, f"cross_rat_geometry_{timestamp}.png")

    results.to_csv(results_path, index=False)
    shuffles.to_csv(shuffles_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_cross_rat_results(results, stats, plot_path)

    print(f"Cross-rat results saved to {results_path}")
    print(f"Cross-rat shuffle controls saved to {shuffles_path}")
    print(f"Cross-rat stats saved to {stats_path}")
    print(f"Cross-rat plot saved to {plot_path}")
    print("Cross-rat stats by comparison family:")
    for _, row in stats.iterrows():
        print(
            f"  {row['comparison_family']} ({row['stats_unit']}): n={int(row['n_runs'])}, "
            f"rReal={row['real_mean']:.6f} +/- {row['real_sem']:.6f}, "
            f"rShuff={row['shuff_mean']:.6f} +/- {row['shuff_sem']:.6f}, "
            f"diff={row['diff_mean']:.6f} +/- {row['diff_sem']:.6f}, "
            f"sign-flip p={row['sign_flip_p_two_sided']:.6g}, "
            f"paired t p={row['paired_t_p_two_sided']:.6g}"
        )
    print(
        "Note: result rows are run-level rat-pair comparisons, and stats are reported "
        "both over runs and over rat-pair means. Individual pairwise distances within "
        "a task-distance matrix are never treated as independent samples."
    )


if __name__ == "__main__":
    main()
