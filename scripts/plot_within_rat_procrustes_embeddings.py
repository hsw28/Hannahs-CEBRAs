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


def build_within_rat_table(rat_files, zscore_before_align=True):
    rows = []
    for path in rat_files:
        rat = load_mean_embeddings(path)
        z_a = rat["A"]
        z_b = rat["B"]
        if zscore_before_align:
            z_a = zscore_columns(z_a)
            z_b = zscore_columns(z_b)

        z_b_aligned = procrustes_align(z_b, z_a)

        for env, embedding in [("A_reference", z_a), ("B_procrustes_to_A", z_b_aligned)]:
            for bin_value, coords in zip(rat["bins"], embedding):
                row = {
                    "rat_id": rat["rat_id"],
                    "environment": env,
                    "task_bin": bin_value,
                    "x": coords[0],
                    "y": coords[1] if embedding.shape[1] > 1 else np.nan,
                    "z": coords[2] if embedding.shape[1] > 2 else np.nan,
                }
                for dim_idx, value in enumerate(coords):
                    row[f"dim{dim_idx + 1}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def set_equal_2d_limits(ax, points):
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.nanmax(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius == 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_one_rat(ax, subset, rat_id):
    a = subset[subset["environment"] == "A_reference"].sort_values("task_bin")
    b = subset[subset["environment"] == "B_procrustes_to_A"].sort_values("task_bin")

    for idx, (_, a_row) in enumerate(a.iterrows()):
        b_row = b[b["task_bin"] == a_row["task_bin"]]
        if b_row.empty:
            continue
        b_row = b_row.iloc[0]
        ax.plot(
            [a_row["x"], b_row["x"]],
            [a_row["y"], b_row["y"]],
            color="#c7c7c7",
            linewidth=0.9,
            alpha=0.8,
            label="matched task bins" if idx == 0 else None,
        )

    ax.plot(
        a["x"],
        a["y"],
        color="#2563eb",
        marker="o",
        linestyle="-",
        linewidth=3.0,
        markersize=7.0,
        markerfacecolor="#2563eb",
        markeredgecolor="#2563eb",
        label="A",
    )
    ax.plot(
        b["x"],
        b["y"],
        color="#c2410c",
        marker="o",
        linestyle="--",
        linewidth=3.0,
        markersize=7.0,
        markerfacecolor="white",
        markeredgecolor="#c2410c",
        markeredgewidth=1.8,
        label="B aligned to A",
    )

    for _, row in a.iterrows():
        ax.text(row["x"], row["y"], f" {row['task_bin']:g}", color="#111111", fontsize=8)

    if len(a) >= 2:
        start = a.iloc[0]
        prev = a.iloc[-2]
        end = a.iloc[-1]
        ax.annotate("start", (start["x"], start["y"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.annotate("end", (end["x"], end["y"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.annotate(
            "",
            xy=(end["x"], end["y"]),
            xytext=(prev["x"], prev["y"]),
            arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 1.6},
        )

    points = pd.concat([a[["x", "y"]], b[["x", "y"]]]).to_numpy(dtype=float)
    set_equal_2d_limits(ax, points)
    ax.set_title(f"Representative within-rat Procrustes alignment\n{rat_id}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    handles, labels = ax.get_legend_handles_labels()
    order = ["A", "B aligned to A", "matched task bins"]
    ordered_handles = []
    ordered_labels = []
    for wanted in order:
        for handle, label in zip(handles, labels):
            if label == wanted:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break
    ax.legend(ordered_handles, ordered_labels, frameon=False, fontsize=8, loc="best")


def save_individual_plots(aligned, output_dir, timestamp):
    paths = []
    for rat_id in pd.unique(aligned["rat_id"]):
        subset = aligned[aligned["rat_id"] == rat_id]
        fig = plt.figure(figsize=(5.4, 4.8))
        ax = fig.add_subplot(111)
        plot_one_rat(ax, subset, rat_id)
        fig.tight_layout()
        svg_path = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.svg")
        png_path = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.png")
        pdf_path = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.pdf")
        fig.savefig(svg_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        paths.append((svg_path, png_path, pdf_path))
    return paths


def save_combined_plot(aligned, output_dir, timestamp):
    rats = list(pd.unique(aligned["rat_id"]))
    n_cols = min(3, len(rats))
    n_rows = int(np.ceil(len(rats) / n_cols))
    fig = plt.figure(figsize=(5.0 * n_cols, 4.5 * n_rows))

    for idx, rat_id in enumerate(rats, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        subset = aligned[aligned["rat_id"] == rat_id]
        plot_one_rat(ax, subset, rat_id)

    fig.suptitle("Within-Rat Procrustes Alignment: B to A (2D)", y=0.98)
    fig.tight_layout()
    svg_path = os.path.join(output_dir, f"within_rat_procrustes_2d_combined_{timestamp}.svg")
    png_path = os.path.join(output_dir, f"within_rat_procrustes_2d_combined_{timestamp}.png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Procrustes-align environment B to environment A within each rat and plot task-bin embeddings."
    )
    parser.add_argument("rat_npz", nargs="+", help="Per-rat geometry_preservation *.npz files.")
    parser.add_argument("--output_dir", default="within_rat_procrustes_embeddings", help="Directory for aligned outputs.")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before alignment.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aligned = build_within_rat_table(args.rat_npz, zscore_before_align=not args.no_zscore)

    csv_path = os.path.join(args.output_dir, f"within_rat_procrustes_aligned_{timestamp}.csv")
    aligned.to_csv(csv_path, index=False)
    individual_paths = save_individual_plots(aligned, args.output_dir, timestamp)
    combined_svg, combined_png = save_combined_plot(aligned, args.output_dir, timestamp)

    print(f"Aligned coordinates saved to {csv_path}")
    print(f"Combined SVG saved to {combined_svg}")
    print(f"Combined PNG saved to {combined_png}")
    for svg_path, png_path, pdf_path in individual_paths:
        print(f"Individual SVG saved to {svg_path}")
        print(f"Individual PNG saved to {png_path}")
        print(f"Individual PDF saved to {pdf_path}")
    print("Environment A is the within-rat reference; environment B is Procrustes-aligned to A.")


if __name__ == "__main__":
    main()
