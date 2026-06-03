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


def alignment_metrics(source, target, aligned):
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


def build_aligned_outputs(rat_files, reference_index=0, reference_env="A", method="similarity", zscore_before_align=True):
    rats = [load_mean_embeddings(path) for path in rat_files]
    reference = rats[reference_index][reference_env]
    if zscore_before_align:
        reference = zscore_columns(reference)

    rows = []
    metric_rows = []
    for rat in rats:
        for env in ["A", "B"]:
            embedding = rat[env]
            if zscore_before_align:
                embedding = zscore_columns(embedding)
            aligned = align_embedding(embedding, reference, method)
            metric_rows.append(
                {
                    "rat_id": rat["rat_id"],
                    "environment": env,
                    "reference_rat_id": rats[reference_index]["rat_id"],
                    "reference_env": reference_env,
                    "alignment_method": method,
                    "zscore_before_align": zscore_before_align,
                    **alignment_metrics(embedding, reference, aligned),
                }
            )

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

    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def build_aligned_table(rat_files, reference_index=0, reference_env="A", method="similarity", zscore_before_align=True):
    aligned, _ = build_aligned_outputs(
        rat_files,
        reference_index=reference_index,
        reference_env=reference_env,
        method=method,
        zscore_before_align=zscore_before_align,
    )
    return aligned


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
    parser.add_argument("rat_npz", nargs="*", help="Per-rat geometry_preservation *.npz files.")
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Directory containing geometry_preservation_rat*.npz files. When provided, "
            "the newest non-checkpoint NPZ is selected for each rat."
        ),
    )
    parser.add_argument("--output_dir", default="aligned_cross_rat_embeddings", help="Directory for aligned outputs.")
    parser.add_argument("--method", choices=["similarity", "affine"], default="similarity", help="Alignment method.")
    parser.add_argument("--reference_index", type=int, default=0, help="Index of reference rat file.")
    parser.add_argument("--reference_env", choices=["A", "B"], default="A", help="Reference environment.")
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

    aligned, metrics = build_aligned_outputs(
        rat_npz_paths,
        reference_index=args.reference_index,
        reference_env=args.reference_env,
        method=args.method,
        zscore_before_align=not args.no_zscore,
    )

    csv_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.csv")
    metrics_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_disparity_{timestamp}.csv")
    png_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.png")
    svg_path = os.path.join(args.output_dir, f"aligned_cross_rat_embeddings_{args.method}_{timestamp}.svg")

    aligned.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    title = f"Aligned Cross-Rat CEBRA Task Embeddings ({args.method})"
    plot_aligned_embeddings(aligned, png_path, title=title)
    plot_aligned_embeddings(aligned, svg_path, title=title)

    print(f"Aligned coordinates saved to {csv_path}")
    print(f"Alignment disparity metrics saved to {metrics_path}")
    print(f"Aligned PNG saved to {png_path}")
    print(f"Aligned SVG saved to {svg_path}")
    print("Solid lines are environment A; dashed lines are environment B.")


if __name__ == "__main__":
    main()
