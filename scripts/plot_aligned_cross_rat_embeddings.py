import argparse
import os
import sys
from datetime import datetime

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def load_mean_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(f"{npz_path} does not contain zA_runs/zB_runs.")

    rat_id = str(data["rat_id"]) if "rat_id" in data else ""
    if not rat_id:
        rat_id = os.path.basename(npz_path).split("_")[2]

    return {
        "rat_id": rat_id,
        "path": npz_path,
        "bins": np.asarray(data["bins"]),
        "A": np.nanmean(np.asarray(data["zA_runs"], dtype=float), axis=0),
        "B": np.nanmean(np.asarray(data["zB_runs"], dtype=float), axis=0),
    }


def zscore_columns(values):
    values = np.asarray(values, dtype=float)
    mu = np.mean(values, axis=0, keepdims=True)
    sigma = np.std(values, axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (values - mu) / sigma


def align_similarity(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_center = np.mean(source, axis=0, keepdims=True)
    target_center = np.mean(target, axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center

    rotation, scale = orthogonal_procrustes(source_centered, target_centered)
    aligned = source_centered @ rotation
    denom = np.sum(aligned ** 2)
    if denom > 0:
        aligned = aligned * (np.sum(aligned * target_centered) / denom)
    return aligned + target_center


def align_affine(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    design = np.hstack([source, np.ones((source.shape[0], 1))])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return design @ coefficients


def align_embedding(source, target, method):
    if method == "similarity":
        return align_similarity(source, target)
    if method == "affine":
        return align_affine(source, target)
    raise ValueError(f"Unknown alignment method: {method}")


def build_aligned_table(rat_files, reference_index=0, reference_env="A", method="similarity", zscore_before_align=True):
    rats = [load_mean_embeddings(path) for path in rat_files]
    reference = rats[reference_index][reference_env]
    if zscore_before_align:
        reference = zscore_columns(reference)

    rows = []
    for rat in rats:
        for env in ["A", "B"]:
            embedding = rat[env]
            if zscore_before_align:
                embedding = zscore_columns(embedding)
            aligned = align_embedding(embedding, reference, method)

            for bin_value, coords in zip(rat["bins"], aligned):
                row = {
                    "rat_id": rat["rat_id"],
                    "environment": env,
                    "task_bin": bin_value,
                    "x": coords[0],
                    "y": coords[1] if aligned.shape[1] > 1 else np.nan,
                    "z": coords[2] if aligned.shape[1] > 2 else np.nan,
                }
                for dim_idx, value in enumerate(coords):
                    row[f"dim{dim_idx + 1}"] = value
                rows.append(row)

    return pd.DataFrame(rows)


def plot_aligned_embeddings(aligned_table, output_path, title=None):
    dims = [col for col in ["x", "y", "z"] if aligned_table[col].notna().any()]
    rats = list(pd.unique(aligned_table["rat_id"]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rats), 3)))
    color_map = {rat: colors[idx] for idx, rat in enumerate(rats)}
    env_styles = {
        "A": {"linestyle": "-", "marker": "o", "alpha": 0.95},
        "B": {"linestyle": "--", "marker": "^", "alpha": 0.85},
    }

    fig = plt.figure(figsize=(7.2, 6.0))
    if len(dims) >= 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    for rat in rats:
        for env in ["A", "B"]:
            subset = aligned_table[(aligned_table["rat_id"] == rat) & (aligned_table["environment"] == env)]
            subset = subset.sort_values("task_bin")
            style = env_styles[env]
            label = f"{rat} {env}"
            if len(dims) >= 3:
                ax.plot(
                    subset["x"],
                    subset["y"],
                    subset["z"],
                    color=color_map[rat],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    alpha=style["alpha"],
                    linewidth=1.6,
                    markersize=5,
                    label=label,
                )
            else:
                ax.plot(
                    subset["x"],
                    subset["y"],
                    color=color_map[rat],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    alpha=style["alpha"],
                    linewidth=1.6,
                    markersize=5,
                    label=label,
                )

    ax.set_xlabel("Aligned dim 1")
    ax.set_ylabel("Aligned dim 2")
    if len(dims) >= 3:
        ax.set_zlabel("Aligned dim 3")
    ax.set_title(title or "Aligned Cross-Rat CEBRA Task Embeddings")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Align and plot per-rat CEBRA task-bin embeddings from geometry_preservation .npz files."
    )
    parser.add_argument("rat_npz", nargs="+", help="Per-rat geometry_preservation *.npz files.")
    parser.add_argument("--output_dir", default="aligned_cross_rat_embeddings", help="Directory for aligned outputs.")
    parser.add_argument("--method", choices=["similarity", "affine"], default="similarity", help="Alignment method.")
    parser.add_argument("--reference_index", type=int, default=0, help="Index of reference rat file.")
    parser.add_argument("--reference_env", choices=["A", "B"], default="A", help="Reference environment.")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before alignment.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aligned = build_aligned_table(
        args.rat_npz,
        reference_index=args.reference_index,
        reference_env=args.reference_env,
        method=args.method,
        zscore_before_align=not args.no_zscore,
    )

    csv_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.csv")
    png_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.png")
    svg_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.svg")

    aligned.to_csv(csv_path, index=False)
    title = f"Aligned Cross-Rat CEBRA Task Embeddings ({args.method})"
    plot_aligned_embeddings(aligned, png_path, title=title)
    plot_aligned_embeddings(aligned, svg_path, title=title)

    print(f"Aligned coordinates saved to {csv_path}")
    print(f"Aligned PNG saved to {png_path}")
    print(f"Aligned SVG saved to {svg_path}")
    print("Solid lines are environment A; dashed lines are environment B.")


if __name__ == "__main__":
    main()
