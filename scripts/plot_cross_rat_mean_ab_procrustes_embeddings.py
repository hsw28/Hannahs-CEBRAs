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


def build_rat_mean_trajectory_outputs(rat_files, reference_index=0, zscore_before_align=True):
    rats = [load_mean_embeddings(path) for path in rat_files]
    rat_trajectories = []
    metric_rows = []

    for rat in rats:
        z_a = rat["A"]
        z_b = rat["B"]
        if zscore_before_align:
            z_a = zscore_columns(z_a)
            z_b = zscore_columns(z_b)

        z_b_aligned = procrustes_align(z_b, z_a)
        mean_ab = (z_a + z_b_aligned) / 2.0
        metric_rows.append(
            {
                "rat_id": rat["rat_id"],
                "comparison": "within_rat_B_to_A",
                "reference_rat_id": rat["rat_id"],
                "zscore_before_align": zscore_before_align,
                **procrustes_alignment_metrics(z_b, z_a, z_b_aligned),
            }
        )
        rat_trajectories.append({**rat, "mean_ab": mean_ab})

    reference = rat_trajectories[reference_index]["mean_ab"]
    reference_rat_id = rat_trajectories[reference_index]["rat_id"]
    rows = []
    for rat in rat_trajectories:
        if rat is rat_trajectories[reference_index]:
            aligned = rat["mean_ab"]
        else:
            aligned = procrustes_align(rat["mean_ab"], reference)
        metric_rows.append(
            {
                "rat_id": rat["rat_id"],
                "comparison": "cross_rat_mean_ab_to_reference",
                "reference_rat_id": reference_rat_id,
                "zscore_before_align": zscore_before_align,
                **procrustes_alignment_metrics(rat["mean_ab"], reference, aligned),
            }
        )

        for bin_value, coords in zip(rat["bins"], aligned):
            row = {
                "rat_id": rat["rat_id"],
                "task_bin": bin_value,
                "x": coords[0],
                "y": coords[1] if aligned.shape[1] > 1 else np.nan,
                "z": coords[2] if aligned.shape[1] > 2 else np.nan,
            }
            for dim_idx, value in enumerate(coords):
                row[f"dim{dim_idx + 1}"] = value
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def build_rat_mean_trajectories(rat_files, reference_index=0, zscore_before_align=True):
    aligned, _ = build_rat_mean_trajectory_outputs(
        rat_files,
        reference_index=reference_index,
        zscore_before_align=zscore_before_align,
    )
    return aligned


def save_figure(fig, output_path):
    kwargs = {"bbox_inches": "tight"}
    if output_path.endswith(".png"):
        kwargs["dpi"] = 300
    fig.savefig(output_path, **kwargs)


def group_mean_trajectory(aligned):
    return (
        aligned.groupby("task_bin", as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"), z=("z", "mean"))
        .sort_values("task_bin")
    )


def plot_group_mean_3d(aligned, output_path, label_bins=False):
    rats = list(pd.unique(aligned["rat_id"]))
    mean_traj = group_mean_trajectory(aligned)

    fig = plt.figure(figsize=(6.8, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    for rat_idx, rat in enumerate(rats):
        subset = aligned[aligned["rat_id"] == rat].sort_values("task_bin")
        ax.plot(
            subset["x"],
            subset["y"],
            subset["z"],
            color="#b8b8b8",
            marker="o",
            linewidth=0.9,
            markersize=3.4,
            alpha=0.75,
            label="Individual rats" if rat_idx == 0 else None,
        )

    ax.plot(
        mean_traj["x"],
        mean_traj["y"],
        mean_traj["z"],
        color="black",
        marker="o",
        linewidth=3.2,
        markersize=7.2,
        label="Group mean",
    )

    if len(mean_traj) >= 2:
        start = mean_traj.iloc[0]
        prev = mean_traj.iloc[-2]
        end = mean_traj.iloc[-1]
        ax.text(start["x"], start["y"], start["z"], " start", color="black", fontsize=8)
        ax.text(end["x"], end["y"], end["z"], " end", color="black", fontsize=8)
        ax.quiver(
            prev["x"],
            prev["y"],
            prev["z"],
            end["x"] - prev["x"],
            end["y"] - prev["y"],
            end["z"] - prev["z"],
            color="black",
            arrow_length_ratio=0.25,
            linewidth=2.0,
        )

    if label_bins:
        for _, row in mean_traj.iterrows():
            ax.text(row["x"], row["y"], row["z"], f" {row['task_bin']:g}", color="black", fontsize=7)

    ax.set_xlabel("Aligned dim 1")
    ax.set_ylabel("Aligned dim 2")
    ax.set_zlabel("Aligned dim 3")
    ax.set_title("Cross-rat Procrustes alignment of mean task trajectories")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def pca_project_aligned(aligned):
    all_pts = aligned[["x", "y", "z"]].to_numpy(dtype=float)
    mu = np.nanmean(all_pts, axis=0, keepdims=True)
    centered = all_pts - mu
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coeff = vt.T
    eigenvalues = singular_values ** 2 / max(all_pts.shape[0] - 1, 1)
    explained = 100.0 * eigenvalues / np.sum(eigenvalues)

    projected = aligned.copy()
    scores = centered @ coeff[:, :2]
    projected["PC1"] = scores[:, 0]
    projected["PC2"] = scores[:, 1]

    mean_traj = group_mean_trajectory(aligned)
    mean_pts = mean_traj[["x", "y", "z"]].to_numpy(dtype=float)
    mean_scores = (mean_pts - mu) @ coeff[:, :2]
    mean_traj["PC1"] = mean_scores[:, 0]
    mean_traj["PC2"] = mean_scores[:, 1]
    return projected, mean_traj, explained, mu.reshape(-1), coeff


def plot_group_mean_2d(projected, mean_traj, explained, output_path, label_bins=False):
    rats = list(pd.unique(projected["rat_id"]))
    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    for rat_idx, rat in enumerate(rats):
        subset = projected[projected["rat_id"] == rat].sort_values("task_bin")
        ax.plot(
            subset["PC1"],
            subset["PC2"],
            color="#b8b8b8",
            marker="o",
            linewidth=0.9,
            markersize=3.4,
            alpha=0.75,
            label="Individual rats" if rat_idx == 0 else None,
        )

    ax.plot(
        mean_traj["PC1"],
        mean_traj["PC2"],
        color="black",
        marker="o",
        linewidth=3.2,
        markersize=7.2,
        label="Group mean",
    )

    if len(mean_traj) >= 2:
        start = mean_traj.iloc[0]
        prev = mean_traj.iloc[-2]
        end = mean_traj.iloc[-1]
        ax.annotate("start", (start["PC1"], start["PC2"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.annotate("end", (end["PC1"], end["PC2"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.annotate(
            "",
            xy=(end["PC1"], end["PC2"]),
            xytext=(prev["PC1"], prev["PC2"]),
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2.0},
        )

    if label_bins:
        for _, row in mean_traj.iterrows():
            ax.annotate(f"{row['task_bin']:g}", (row["PC1"], row["PC2"]), xytext=(5, 5), textcoords="offset points", fontsize=7)

    top_two = explained[:2].sum()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cross-rat aligned task trajectory")
    ax.text(
        0.02,
        0.98,
        f"First two PCs explain {top_two:.1f}% variance",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Within each rat, Procrustes-align B to A and average A/B; then "
            "Procrustes-align the resulting one-line-per-rat trajectories across rats."
        )
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
    parser.add_argument("--output_dir", default="cross_rat_mean_ab_procrustes_embeddings", help="Output directory.")
    parser.add_argument("--reference_index", type=int, default=0, help="Index of reference rat file.")
    parser.add_argument("--no_zscore", action="store_true", help="Do not z-score embedding dimensions before within-rat alignment.")
    parser.add_argument("--label_bins", action="store_true", help="Label group mean task bins on the plots.")
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

    aligned, metrics = build_rat_mean_trajectory_outputs(
        rat_npz_paths,
        reference_index=args.reference_index,
        zscore_before_align=not args.no_zscore,
    )

    csv_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_{timestamp}.csv")
    metrics_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_disparity_{timestamp}.csv")
    png3d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_group_mean_3d_{timestamp}.png")
    svg3d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_group_mean_3d_{timestamp}.svg")
    pdf3d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_group_mean_3d_{timestamp}.pdf")
    png2d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_pca2d_{timestamp}.png")
    svg2d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_pca2d_{timestamp}.svg")
    pdf2d_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_pca2d_{timestamp}.pdf")
    pca_csv_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_pca2d_scores_{timestamp}.csv")
    pca_info_path = os.path.join(args.output_dir, f"cross_rat_mean_ab_procrustes_pca2d_info_{timestamp}.csv")

    aligned.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_group_mean_3d(aligned, png3d_path, label_bins=args.label_bins)
    plot_group_mean_3d(aligned, svg3d_path, label_bins=args.label_bins)
    plot_group_mean_3d(aligned, pdf3d_path, label_bins=args.label_bins)

    projected, mean_traj, explained, mu, coeff = pca_project_aligned(aligned)
    projected.to_csv(pca_csv_path, index=False)
    pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "explained_percent": explained[:3],
            "mean_xyz": list(mu[:3]),
        }
    ).to_csv(pca_info_path, index=False)
    plot_group_mean_2d(projected, mean_traj, explained, png2d_path, label_bins=args.label_bins)
    plot_group_mean_2d(projected, mean_traj, explained, svg2d_path, label_bins=args.label_bins)
    plot_group_mean_2d(projected, mean_traj, explained, pdf2d_path, label_bins=args.label_bins)

    print(f"Aligned mean A/B coordinates saved to {csv_path}")
    print(f"Procrustes disparity metrics saved to {metrics_path}")
    print(f"3D PNG saved to {png3d_path}")
    print(f"3D SVG saved to {svg3d_path}")
    print(f"3D PDF saved to {pdf3d_path}")
    print(f"2D PCA scores saved to {pca_csv_path}")
    print(f"2D PCA info saved to {pca_info_path}")
    print(f"2D PNG saved to {png2d_path}")
    print(f"2D SVG saved to {svg2d_path}")
    print(f"2D PDF saved to {pdf2d_path}")
    print(f"First two PCs explain {explained[:2].sum():.2f}% variance")
    print("Each line is one rat: B was first Procrustes-aligned to A within rat, then A/B were averaged.")


if __name__ == "__main__":
    main()
