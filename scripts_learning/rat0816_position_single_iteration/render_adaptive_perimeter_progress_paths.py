import argparse
import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D


DATASETS = {
    "train": {
        "embedding_key": "emb_train",
        "position_key": "pos_train",
        "display": "Environment A train",
        "line_width": 4.8,
        "point_alpha": 0.11,
    },
    "holdout": {
        "embedding_key": "emb_holdout",
        "position_key": "pos_holdout",
        "display": "Environment A held out",
        "line_width": 4.0,
        "point_alpha": 0.08,
    },
    "b": {
        "embedding_key": "emb_b",
        "position_key": "pos_b",
        "display": "Environment B",
        "line_width": 5.6,
        "point_alpha": 0.11,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render occupancy-adaptive perimeter-progress paths from an existing Rat 5 position run."
    )
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--render-bins", default=12, type=int)
    parser.add_argument("--tested-bins", default="12,16,20,24", type=str)
    parser.add_argument("--band-fraction", default=0.12, type=float)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--dpi", default=180, type=int)
    parser.add_argument("--elev", default=21.0, type=float)
    parser.add_argument("--azim", default=318.0, type=float)
    return parser.parse_args()


def load_data(run_dir: Path):
    npz = np.load(run_dir / "embeddings_and_labels.npz")
    loaded = {}
    for name, spec in DATASETS.items():
        loaded[name] = {
            "embedding": npz[spec["embedding_key"]],
            "position": npz[spec["position_key"]],
            "display": spec["display"],
            "line_width": spec["line_width"],
            "point_alpha": spec["point_alpha"],
        }
    return loaded


def robust_arena_bounds(loaded):
    all_positions = np.concatenate([item["position"] for item in loaded.values()], axis=0)
    x_min, x_max = np.quantile(all_positions[:, 0], [0.02, 0.98])
    y_min, y_max = np.quantile(all_positions[:, 1], [0.02, 0.98])
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }


def nearest_wall_distance(position: np.ndarray, bounds):
    left = np.abs(position[:, 0] - bounds["x_min"])
    right = np.abs(bounds["x_max"] - position[:, 0])
    bottom = np.abs(position[:, 1] - bounds["y_min"])
    top = np.abs(bounds["y_max"] - position[:, 1])
    distances = np.stack([bottom, right, top, left], axis=1)
    return distances.min(axis=1), distances.argmin(axis=1)


def perimeter_progress(position: np.ndarray, bounds):
    width = bounds["x_max"] - bounds["x_min"]
    height = bounds["y_max"] - bounds["y_min"]
    perimeter = 2.0 * (width + height)
    wall_distance, wall_index = nearest_wall_distance(position, bounds)

    x = np.clip(position[:, 0], bounds["x_min"], bounds["x_max"])
    y = np.clip(position[:, 1], bounds["y_min"], bounds["y_max"])

    progress = np.zeros(len(position), dtype=float)
    bottom_mask = wall_index == 0
    right_mask = wall_index == 1
    top_mask = wall_index == 2
    left_mask = wall_index == 3

    progress[bottom_mask] = (x[bottom_mask] - bounds["x_min"]) / perimeter
    progress[right_mask] = (width + (y[right_mask] - bounds["y_min"])) / perimeter
    progress[top_mask] = (width + height + (bounds["x_max"] - x[top_mask])) / perimeter
    progress[left_mask] = (2.0 * width + height + (bounds["y_max"] - y[left_mask])) / perimeter
    return progress, wall_distance


def compute_band_width(loaded, bounds, band_fraction: float):
    all_positions = np.concatenate([item["position"] for item in loaded.values()], axis=0)
    wall_distance, _ = nearest_wall_distance(all_positions, bounds)
    min_dim = min(bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"])
    return float(max(min_dim * band_fraction, np.quantile(wall_distance, 0.22)))


def compute_limits(loaded):
    stacked = np.concatenate([item["embedding"] for item in loaded.values()], axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.08
    return centers, radius


def style_axis(ax, centers, radius):
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.xaxis.line.set_alpha(0.0)
    ax.yaxis.line.set_alpha(0.0)
    ax.zaxis.line.set_alpha(0.0)
    ax.grid(False)
    ax.set_facecolor("#060606")


def quantile_edges(values: np.ndarray, n_bins: int):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = np.nextafter(edges[idx - 1], 1.0)
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray):
    n_bins = len(edges) - 1
    indices = np.searchsorted(edges, values, side="right") - 1
    return np.clip(indices, 0, n_bins - 1)


def build_adaptive_paths(loaded, edges: np.ndarray):
    n_bins = len(edges) - 1
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    adaptive = {}
    for name in DATASETS:
        values = loaded[name]["progress"][loaded[name]["mask"]]
        indices = assign_bins(values, edges)
        counts = np.bincount(indices, minlength=n_bins)

        centroids = np.full((n_bins, loaded[name]["embedding"].shape[1]), np.nan, dtype=float)
        full_bin_indices = assign_bins(loaded[name]["progress"], edges)
        for idx in range(n_bins):
            take = loaded[name]["mask"] & (full_bin_indices == idx)
            if np.any(take):
                centroids[idx] = loaded[name]["embedding"][take].mean(axis=0)

        adaptive[name] = {
            "counts": counts,
            "centroids": centroids,
            "valid": ~np.isnan(centroids[:, 0]),
            "bin_centers": bin_centers,
            "bin_indices_all": full_bin_indices,
        }
    return adaptive


def compute_contiguous_segments(valid_mask: np.ndarray):
    segments = []
    start = None
    for idx, is_valid in enumerate(valid_mask):
        if is_valid and start is None:
            start = idx
        if (not is_valid) and start is not None:
            if idx - start >= 2:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(valid_mask) - start >= 2:
        segments.append((start, len(valid_mask) - 1))
    return segments


def maybe_wrap_segment(valid_mask: np.ndarray):
    if len(valid_mask) < 2:
        return False
    return bool(valid_mask[0] and valid_mask[-1])


def plot_path_segments(ax, centroids: np.ndarray, bin_centers: np.ndarray, valid_mask: np.ndarray, cmap, line_width: float, alpha: float):
    segments = compute_contiguous_segments(valid_mask)
    for start, end in segments:
        for idx in range(start, end):
            color = cmap(bin_centers[idx])
            ax.plot(
                centroids[idx : idx + 2, 0],
                centroids[idx : idx + 2, 1],
                centroids[idx : idx + 2, 2],
                color=color,
                linewidth=line_width,
                alpha=alpha,
                solid_capstyle="round",
            )
    if maybe_wrap_segment(valid_mask):
        color = cmap(bin_centers[-1])
        ax.plot(
            [centroids[-1, 0], centroids[0, 0]],
            [centroids[-1, 1], centroids[0, 1]],
            [centroids[-1, 2], centroids[0, 2]],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            solid_capstyle="round",
        )


def scatter_points(ax, loaded_item, cmap):
    mask = loaded_item["mask"]
    ax.scatter(
        loaded_item["embedding"][mask, 0],
        loaded_item["embedding"][mask, 1],
        loaded_item["embedding"][mask, 2],
        c=loaded_item["progress"][mask],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=7,
        alpha=loaded_item["point_alpha"],
        edgecolors="none",
        depthshade=False,
    )


def build_points_figure(run_dir: Path, loaded, adaptive, centers, radius, cmap, render_bins: int):
    fig = plt.figure(figsize=(16, 14), facecolor="#060606")
    panel_order = [
        ("train", "A train"),
        ("holdout", "A held out"),
        ("b", "B"),
        ("overlay", "Adaptive centroid-path comparison"),
    ]

    axes = []
    for idx, (key, title) in enumerate(panel_order, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        style_axis(ax, centers, radius)
        ax.set_title(title, color="#f3f3f3", fontsize=12, pad=10)
        axes.append(ax)

        if key == "overlay":
            scatter_points(ax, loaded["train"], cmap)
            scatter_points(ax, loaded["b"], cmap)
            for dataset_name, alpha in [("train", 0.45), ("holdout", 0.28), ("b", 0.98)]:
                plot_path_segments(
                    ax,
                    adaptive[dataset_name]["centroids"],
                    adaptive[dataset_name]["bin_centers"],
                    adaptive[dataset_name]["valid"],
                    cmap,
                    DATASETS[dataset_name]["line_width"],
                    alpha,
                )
            ax.text2D(
                0.03,
                0.92,
                f"B-quantile bins ({render_bins})\nGaps left empty",
                transform=ax.transAxes,
                color="#f3f3f3",
                fontsize=10,
            )
            continue

        scatter_points(ax, loaded[key], cmap)
        plot_path_segments(
            ax,
            adaptive[key]["centroids"],
            adaptive[key]["bin_centers"],
            adaptive[key]["valid"],
            cmap,
            DATASETS[key]["line_width"],
            1.0,
        )

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(0.0, 1.0)
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.022, pad=0.02)
    cbar.set_label("Perimeter progress", color="#f3f3f3", rotation=270, labelpad=22)
    cbar.ax.tick_params(colors="#f3f3f3")
    cbar.outline.set_edgecolor("#d8d8d8")

    fig.text(
        0.5,
        0.03,
        "Occupancy-adaptive bins are defined from B perimeter progress and then applied identically to A and B.",
        ha="center",
        va="center",
        color="#f3f3f3",
        fontsize=12,
    )

    svg_path = run_dir / f"adaptive_quantile_bins{render_bins}_with_points.svg"
    png_path = run_dir / f"adaptive_quantile_bins{render_bins}_with_points.png"
    fig.savefig(svg_path, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


def build_paths_only_figure(run_dir: Path, adaptive, centers, radius, cmap, render_bins: int):
    fig = plt.figure(figsize=(10.5, 9.2), facecolor="#060606")
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius)
    ax.set_title("Occupancy-adaptive perimeter centroid paths", color="#f3f3f3", fontsize=15, pad=12)

    for dataset_name, alpha in [("train", 0.85), ("holdout", 0.65), ("b", 1.0)]:
        plot_path_segments(
            ax,
            adaptive[dataset_name]["centroids"],
            adaptive[dataset_name]["bin_centers"],
            adaptive[dataset_name]["valid"],
            cmap,
            DATASETS[dataset_name]["line_width"],
            alpha,
        )

    handles = [
        Line2D([0], [0], color="#a6a6a6", lw=DATASETS["train"]["line_width"], label="A train"),
        Line2D([0], [0], color="#d0d0d0", lw=DATASETS["holdout"]["line_width"], label="A held out"),
        Line2D([0], [0], color="#ffffff", lw=DATASETS["b"]["line_width"], label="B"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color("#f3f3f3")

    ax.text2D(
        0.03,
        0.05,
        f"B-quantile bins: {render_bins}\nEmpty bins are left empty",
        transform=ax.transAxes,
        color="#f3f3f3",
        fontsize=11,
    )

    svg_path = run_dir / f"adaptive_quantile_bins{render_bins}_paths_only.svg"
    png_path = run_dir / f"adaptive_quantile_bins{render_bins}_paths_only.png"
    fig.savefig(svg_path, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


def render_movie(run_dir: Path, loaded, adaptive, centers, radius, cmap, render_bins: int, fps: int, dpi: int, elev: float, azim: float):
    video_path = run_dir / f"adaptive_quantile_bins{render_bins}_paths.mp4"
    preview_path = run_dir / f"adaptive_quantile_bins{render_bins}_paths_preview.png"

    fig = plt.figure(figsize=(8.8, 8.8), facecolor="#060606")
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius)
    ax.view_init(elev=elev, azim=azim)

    title = ax.set_title("", color="#f3f3f3", fontsize=15, pad=10)
    subtitle = fig.text(0.5, 0.06, "", ha="center", va="center", color="#f3f3f3", fontsize=14)

    scatter_artists = {}
    segment_artists = {}
    for name in DATASETS:
        mask = loaded[name]["mask"]
        scatter_artists[name] = ax.scatter(
            loaded[name]["embedding"][mask, 0],
            loaded[name]["embedding"][mask, 1],
            loaded[name]["embedding"][mask, 2],
            c=loaded[name]["progress"][mask],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=9,
            alpha=0.0,
            edgecolors="none",
            depthshade=False,
        )
        segment_artists[name] = []
        centers_arr = adaptive[name]["centroids"]
        valid = adaptive[name]["valid"]
        for idx in range(len(centers_arr) - 1):
            if not (valid[idx] and valid[idx + 1]):
                segment_artists[name].append(None)
                continue
            color = cmap(adaptive[name]["bin_centers"][idx])
            (line,) = ax.plot([], [], [], color=color, linewidth=DATASETS[name]["line_width"], alpha=0.0, solid_capstyle="round")
            segment_artists[name].append(line)

    wrap_artists = {}
    for name in DATASETS:
        valid = adaptive[name]["valid"]
        if valid[0] and valid[-1]:
            color = cmap(adaptive[name]["bin_centers"][-1])
            (line,) = ax.plot([], [], [], color=color, linewidth=DATASETS[name]["line_width"], alpha=0.0, solid_capstyle="round")
            wrap_artists[name] = line
        else:
            wrap_artists[name] = None

    intro = 18
    draw_frames = 96
    hold = 18
    final_hold = 42
    windows = {
        "train": (intro, intro + draw_frames),
        "holdout": (intro + draw_frames + hold, intro + 2 * draw_frames + hold),
        "b": (intro + 2 * draw_frames + 2 * hold, intro + 3 * draw_frames + 2 * hold),
    }
    total_frames = windows["b"][1] + final_hold

    writer = FFMpegWriter(fps=fps, metadata={"title": f"Adaptive quantile perimeter paths {render_bins}"}, bitrate=9000)
    with writer.saving(fig, str(video_path), dpi):
        for frame_idx in range(total_frames):
            if frame_idx < windows["train"][0]:
                title.set_text("Shared perimeter progress")
                subtitle.set_text("B-quantile adaptive bins; same perimeter definition in A and B")
            elif frame_idx < windows["holdout"][0]:
                title.set_text("Environment A train")
                subtitle.set_text("Adaptive centroid path revealed in perimeter order")
            elif frame_idx < windows["b"][0]:
                title.set_text("Environment A held out")
                subtitle.set_text("Empty bins are left as real gaps")
            elif frame_idx < windows["b"][1]:
                title.set_text("Environment B")
                subtitle.set_text("B defines the occupancy-adaptive perimeter bins")
            else:
                title.set_text("Adaptive A vs B perimeter-path comparison")
                subtitle.set_text("Same route variable, different latent geometry across contexts")

            for name in DATASETS:
                start, end = windows[name]
                if frame_idx < start:
                    progress = 0.0
                elif frame_idx >= end:
                    progress = 1.0
                else:
                    progress = (frame_idx - start + 1) / max(1, end - start)

                if progress == 0.0:
                    scatter_artists[name].set_alpha(0.0)
                else:
                    settled = 0.14 if name != "holdout" else 0.10
                    if frame_idx >= windows["b"][1] and name in {"train", "holdout"}:
                        settled = 0.06 if name == "train" else 0.03
                    scatter_artists[name].set_alpha(settled)

                valid_indices = np.where(adaptive[name]["valid"])[0]
                reveal_count = int(np.floor(progress * len(valid_indices)))
                visible_bins = set(valid_indices[:reveal_count].tolist())

                for seg_idx, line in enumerate(segment_artists[name]):
                    if line is None:
                        continue
                    if seg_idx in visible_bins and (seg_idx + 1) in visible_bins:
                        alpha = 0.98
                        if frame_idx >= windows["b"][1] and name in {"train", "holdout"}:
                            alpha = 0.22 if name == "train" else 0.12
                        line.set_alpha(alpha)
                        line.set_data(
                            adaptive[name]["centroids"][seg_idx : seg_idx + 2, 0],
                            adaptive[name]["centroids"][seg_idx : seg_idx + 2, 1],
                        )
                        line.set_3d_properties(adaptive[name]["centroids"][seg_idx : seg_idx + 2, 2])
                    else:
                        line.set_alpha(0.0)
                        line.set_data([], [])
                        line.set_3d_properties([])

                wrap_line = wrap_artists[name]
                if wrap_line is not None and adaptive[name]["valid"][0] and adaptive[name]["valid"][-1] and 0 in visible_bins and (len(adaptive[name]["valid"]) - 1) in visible_bins:
                    alpha = 0.98
                    if frame_idx >= windows["b"][1] and name in {"train", "holdout"}:
                        alpha = 0.22 if name == "train" else 0.12
                    wrap_line.set_alpha(alpha)
                    wrap_line.set_data(
                        [adaptive[name]["centroids"][-1, 0], adaptive[name]["centroids"][0, 0]],
                        [adaptive[name]["centroids"][-1, 1], adaptive[name]["centroids"][0, 1]],
                    )
                    wrap_line.set_3d_properties(
                        [adaptive[name]["centroids"][-1, 2], adaptive[name]["centroids"][0, 2]]
                    )
                elif wrap_line is not None:
                    wrap_line.set_alpha(0.0)
                    wrap_line.set_data([], [])
                    wrap_line.set_3d_properties([])

            if frame_idx == windows["b"][1]:
                fig.savefig(preview_path, dpi=dpi, facecolor=fig.get_facecolor())

            writer.grab_frame()

    plt.close(fig)
    return video_path, preview_path, windows, total_frames


def tested_bin_summary(loaded, tested_bins):
    summary = {}
    for n_bins in tested_bins:
        edges = quantile_edges(loaded["b"]["progress"][loaded["b"]["mask"]], n_bins)
        adaptive = build_adaptive_paths(loaded, edges)
        summary[str(n_bins)] = {
            "requested_bins": n_bins,
            "b_counts": adaptive["b"]["counts"].astype(int).tolist(),
            "train_empty_bins": np.where(adaptive["train"]["counts"] == 0)[0].astype(int).tolist(),
            "holdout_empty_bins": np.where(adaptive["holdout"]["counts"] == 0)[0].astype(int).tolist(),
            "b_empty_bins": np.where(adaptive["b"]["counts"] == 0)[0].astype(int).tolist(),
            "interpolation_needed": False,
        }
    return summary


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    tested_bins = [int(item) for item in args.tested_bins.split(",") if item.strip()]

    loaded = load_data(run_dir)
    bounds = robust_arena_bounds(loaded)
    band_width = compute_band_width(loaded, bounds, args.band_fraction)

    for name in DATASETS:
        progress, wall_distance = perimeter_progress(loaded[name]["position"], bounds)
        loaded[name]["progress"] = progress
        loaded[name]["wall_distance"] = wall_distance
        loaded[name]["mask"] = wall_distance <= band_width

    summary = tested_bin_summary(loaded, tested_bins)

    render_bins = args.render_bins
    edges = quantile_edges(loaded["b"]["progress"][loaded["b"]["mask"]], render_bins)
    adaptive = build_adaptive_paths(loaded, edges)
    centers, radius = compute_limits(loaded)
    cmap = colormaps.get_cmap("turbo")

    points_svg, points_png = build_points_figure(run_dir, loaded, adaptive, centers, radius, cmap, render_bins)
    paths_svg, paths_png = build_paths_only_figure(run_dir, adaptive, centers, radius, cmap, render_bins)
    video_path, preview_path, windows, total_frames = render_movie(
        run_dir,
        loaded,
        adaptive,
        centers,
        radius,
        cmap,
        render_bins,
        fps=args.fps,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
    )

    np.savez_compressed(
        run_dir / f"adaptive_quantile_bins{render_bins}_artifacts.npz",
        edges=edges,
        train_counts=adaptive["train"]["counts"],
        train_valid=adaptive["train"]["valid"],
        train_centroids=adaptive["train"]["centroids"],
        holdout_counts=adaptive["holdout"]["counts"],
        holdout_valid=adaptive["holdout"]["valid"],
        holdout_centroids=adaptive["holdout"]["centroids"],
        b_counts=adaptive["b"]["counts"],
        b_valid=adaptive["b"]["valid"],
        b_centroids=adaptive["b"]["centroids"],
    )

    metadata = {
        "run_dir": str(run_dir),
        "render_bins": render_bins,
        "tested_bins": tested_bins,
        "arena_bounds": bounds,
        "band_width": band_width,
        "band_fraction": args.band_fraction,
        "movie_windows": windows,
        "total_frames": total_frames,
        "tested_bin_summary": summary,
        "artifacts": {
            "with_points_svg": str(points_svg),
            "with_points_png": str(points_png),
            "paths_only_svg": str(paths_svg),
            "paths_only_png": str(paths_png),
            "video_mp4": str(video_path),
            "video_preview_png": str(preview_path),
            "adaptive_npz": str(run_dir / f"adaptive_quantile_bins{render_bins}_artifacts.npz"),
        },
        "notes": {
            "perimeter_definition": "Same clockwise rectangular perimeter from pooled A/B bounds and the same lower-left start corner.",
            "adaptive_bin_definition": "Quantile edges are defined from B perimeter progress within the shared perimeter band, then applied identically to A train, A held-out, and B.",
            "gap_policy": "Empty bins are left empty; no interpolation is used in this adaptive-quantile version.",
        },
    }
    summary_path = run_dir / f"adaptive_quantile_bins{render_bins}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved adaptive points figure to {points_svg}")
    print(f"Saved adaptive paths-only figure to {paths_svg}")
    print(f"Saved adaptive movie to {video_path}")
    print(f"Saved adaptive summary to {summary_path}")


if __name__ == "__main__":
    main()
