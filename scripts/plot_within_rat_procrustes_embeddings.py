import argparse
import glob
import os
import re
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
from scipy.spatial import procrustes


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
    return os.path.basename(npz_path).split("_")[2]


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


def load_mean_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "zA_runs" not in data or "zB_runs" not in data:
        raise ValueError(f"{npz_path} does not contain zA_runs/zB_runs.")

    return {
        "rat_id": load_rat_id(data, npz_path),
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


def procrustes_alignment_metrics(source, target, aligned):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    aligned = np.asarray(aligned, dtype=float)

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
        "n_points": target.shape[0],
        "n_dimensions": target.shape[1],
        "procrustes_disparity": disparity,
        "aligned_sse": aligned_sse,
        "aligned_rmse": aligned_rmse,
        "target_total_ss": target_total_ss,
        "aligned_sse_over_target_ss": aligned_sse_over_target_ss,
    }


def build_within_rat_outputs(rat_files, zscore_before_align=True):
    rows = []
    metric_rows = []
    for path in rat_files:
        rat = load_mean_embeddings(path)
        z_a = rat["A"]
        z_b = rat["B"]
        if zscore_before_align:
            z_a = zscore_columns(z_a)
            z_b = zscore_columns(z_b)

        z_b_aligned = procrustes_align(z_b, z_a)
        metric_rows.append(
            {
                "rat_id": rat["rat_id"],
                "source_npz": path,
                "comparison": "B_to_A",
                "zscore_before_align": zscore_before_align,
                **procrustes_alignment_metrics(z_b, z_a, z_b_aligned),
            }
        )

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
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def build_within_rat_table(rat_files, zscore_before_align=True):
    aligned, _ = build_within_rat_outputs(rat_files, zscore_before_align=zscore_before_align)
    return aligned


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


def set_equal_3d_limits(ax, points):
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.nanmax(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius == 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.55)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.0))
        axis.pane.set_edgecolor("#eeeeee")


def get_rat_trajectories(subset):
    a = subset[subset["environment"] == "A_reference"].sort_values("task_bin")
    b = subset[subset["environment"] == "B_procrustes_to_A"].sort_values("task_bin")
    return a, b


def plot_one_rat_2d(ax, subset, rat_id):
    a, b = get_rat_trajectories(subset)
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


def plot_one_rat_3d(ax, subset, rat_id):
    a, b = get_rat_trajectories(subset)

    for idx, (_, a_row) in enumerate(a.iterrows()):
        b_row = b[b["task_bin"] == a_row["task_bin"]]
        if b_row.empty:
            continue
        b_row = b_row.iloc[0]
        ax.plot(
            [a_row["x"], b_row["x"]],
            [a_row["y"], b_row["y"]],
            [a_row["z"], b_row["z"]],
            color="#c7c7c7",
            linewidth=0.8,
            alpha=0.75,
            label="matched task bins" if idx == 0 else None,
        )

    ax.plot(
        a["x"],
        a["y"],
        a["z"],
        color="#2563eb",
        marker="o",
        linestyle="-",
        linewidth=3.0,
        markersize=6.8,
        markerfacecolor="#2563eb",
        markeredgecolor="#2563eb",
        label="A",
    )
    ax.plot(
        b["x"],
        b["y"],
        b["z"],
        color="#c2410c",
        marker="o",
        linestyle="--",
        linewidth=3.0,
        markersize=6.8,
        markerfacecolor="white",
        markeredgecolor="#c2410c",
        markeredgewidth=1.7,
        label="B aligned to A",
    )

    for _, row in a.iterrows():
        ax.text(row["x"], row["y"], row["z"], f" {row['task_bin']:g}", color="#111111", fontsize=8)

    if len(a) >= 2:
        start = a.iloc[0]
        prev = a.iloc[-2]
        end = a.iloc[-1]
        ax.text(start["x"], start["y"], start["z"], " start", fontsize=8, color="#111111")
        ax.text(end["x"], end["y"], end["z"], " end", fontsize=8, color="#111111")
        ax.quiver(
            prev["x"],
            prev["y"],
            prev["z"],
            end["x"] - prev["x"],
            end["y"] - prev["y"],
            end["z"] - prev["z"],
            color="#111111",
            linewidth=1.3,
            arrow_length_ratio=0.25,
            length=1.0,
            normalize=False,
        )

    points = pd.concat([a[["x", "y", "z"]], b[["x", "y", "z"]]]).to_numpy(dtype=float)
    set_equal_3d_limits(ax, points)
    ax.view_init(elev=22, azim=-58)
    ax.set_title(f"Representative within-rat Procrustes alignment\n{rat_id}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
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
        plot_one_rat_2d(ax, subset, rat_id)
        fig.tight_layout()
        svg_2d = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.svg")
        png_2d = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.png")
        pdf_2d = os.path.join(output_dir, f"within_rat_procrustes_2d_{rat_id}_{timestamp}.pdf")
        fig.savefig(svg_2d, bbox_inches="tight")
        fig.savefig(png_2d, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_2d, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(5.8, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        plot_one_rat_3d(ax, subset, rat_id)
        fig.tight_layout()
        svg_3d = os.path.join(output_dir, f"within_rat_procrustes_3d_{rat_id}_{timestamp}.svg")
        png_3d = os.path.join(output_dir, f"within_rat_procrustes_3d_{rat_id}_{timestamp}.png")
        pdf_3d = os.path.join(output_dir, f"within_rat_procrustes_3d_{rat_id}_{timestamp}.pdf")
        fig.savefig(svg_3d, bbox_inches="tight")
        fig.savefig(png_3d, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_3d, bbox_inches="tight")
        plt.close(fig)

        paths.append(
            {
                "rat_id": rat_id,
                "2d": (svg_2d, png_2d, pdf_2d),
                "3d": (svg_3d, png_3d, pdf_3d),
            }
        )
    return paths


def save_combined_plot(aligned, output_dir, timestamp):
    rats = list(pd.unique(aligned["rat_id"]))
    n_cols = min(3, len(rats))
    n_rows = int(np.ceil(len(rats) / n_cols))
    fig = plt.figure(figsize=(5.0 * n_cols, 4.5 * n_rows))

    for idx, rat_id in enumerate(rats, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        subset = aligned[aligned["rat_id"] == rat_id]
        plot_one_rat_2d(ax, subset, rat_id)

    fig.suptitle("Within-Rat Procrustes Alignment: B to A (2D)", y=0.98)
    fig.tight_layout()
    svg_2d = os.path.join(output_dir, f"within_rat_procrustes_2d_combined_{timestamp}.svg")
    png_2d = os.path.join(output_dir, f"within_rat_procrustes_2d_combined_{timestamp}.png")
    fig.savefig(svg_2d, bbox_inches="tight")
    fig.savefig(png_2d, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(5.2 * n_cols, 4.9 * n_rows))
    for idx, rat_id in enumerate(rats, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        subset = aligned[aligned["rat_id"] == rat_id]
        plot_one_rat_3d(ax, subset, rat_id)

    fig.suptitle("Within-Rat Procrustes Alignment: B to A (3D)", y=0.98)
    fig.tight_layout()
    svg_3d = os.path.join(output_dir, f"within_rat_procrustes_3d_combined_{timestamp}.svg")
    png_3d = os.path.join(output_dir, f"within_rat_procrustes_3d_combined_{timestamp}.png")
    fig.savefig(svg_3d, bbox_inches="tight")
    fig.savefig(png_3d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"2d": (svg_2d, png_2d), "3d": (svg_3d, png_3d)}


def main():
    parser = argparse.ArgumentParser(
        description="Procrustes-align environment B to environment A within each rat and plot task-bin embeddings."
    )
    parser.add_argument("rat_npz", nargs="*", help="Per-rat geometry_preservation *.npz files.")
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory containing geometry_preservation_rat*.npz files. When provided, "
            "the newest non-checkpoint NPZ is selected for each rat."
        ),
    )
    parser.add_argument("--output_dir", default="within_rat_procrustes_embeddings", help="Directory for aligned outputs.")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before alignment.")
    parser.add_argument(
        "--require_comparison_mode",
        default="any",
        help=(
            "Optional required comparison_mode in each NPZ. Defaults to 'any'. "
            f"Use '{FULL_POP_COMPARISON_MODE}' to require the full-population A(n)-vs-B(1) mode."
        ),
    )
    args = parser.parse_args()
    required_comparison_mode = None if args.require_comparison_mode == "any" else args.require_comparison_mode
    if bool(args.input_dir) == bool(args.rat_npz):
        raise ValueError("Provide either --input_dir or explicit rat_npz paths, but not both.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    aligned, metrics = build_within_rat_outputs(rat_npz_paths, zscore_before_align=not args.no_zscore)

    csv_path = os.path.join(args.output_dir, f"within_rat_procrustes_aligned_{timestamp}.csv")
    metrics_path = os.path.join(args.output_dir, f"within_rat_procrustes_disparity_{timestamp}.csv")
    aligned.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    individual_paths = save_individual_plots(aligned, args.output_dir, timestamp)
    combined_paths = save_combined_plot(aligned, args.output_dir, timestamp)

    print(f"Aligned coordinates saved to {csv_path}")
    print(f"Procrustes disparity metrics saved to {metrics_path}")
    for view_name, (svg_path, png_path) in combined_paths.items():
        print(f"Combined {view_name.upper()} SVG saved to {svg_path}")
        print(f"Combined {view_name.upper()} PNG saved to {png_path}")
    for path_set in individual_paths:
        rat_id = path_set["rat_id"]
        for view_name in ["2d", "3d"]:
            svg_path, png_path, pdf_path = path_set[view_name]
            if view_name == "rat_id":
                continue
            print(f"Individual {view_name.upper()} SVG for {rat_id} saved to {svg_path}")
            print(f"Individual {view_name.upper()} PNG for {rat_id} saved to {png_path}")
            print(f"Individual {view_name.upper()} PDF for {rat_id} saved to {pdf_path}")
    print("Environment A is the within-rat reference; environment B is Procrustes-aligned to A.")


if __name__ == "__main__":
    main()
