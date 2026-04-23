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


DATASETS = {
    "train": {
        "embedding_key": "emb_train",
        "position_key": "pos_train",
        "display": "Environment A train",
        "line_width": 4.8,
        "path_alpha": 1.0,
        "point_alpha": 0.12,
    },
    "holdout": {
        "embedding_key": "emb_holdout",
        "position_key": "pos_holdout",
        "display": "Environment A held out",
        "line_width": 4.0,
        "path_alpha": 0.92,
        "point_alpha": 0.09,
    },
    "b": {
        "embedding_key": "emb_b",
        "position_key": "pos_b",
        "display": "Environment B",
        "line_width": 5.6,
        "path_alpha": 1.0,
        "point_alpha": 0.12,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render perimeter-progress static figures and a presentation-style path movie from an existing Rat 5 position run."
    )
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--bins", default=48, type=int)
    parser.add_argument("--band-fraction", default=0.12, type=float)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--dpi", default=180, type=int)
    parser.add_argument("--elev", default=21.0, type=float)
    parser.add_argument("--azim", default=318.0, type=float)
    parser.add_argument("--roll", default=0.0, type=float)
    parser.add_argument("--proj-type", default="persp", choices=["persp", "ortho"])
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
            "path_alpha": spec["path_alpha"],
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


def compute_perimeter_band(loaded, bounds, band_fraction: float):
    all_positions = np.concatenate([item["position"] for item in loaded.values()], axis=0)
    wall_distance, _ = nearest_wall_distance(all_positions, bounds)
    min_dim = min(bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"])
    fraction_band = min_dim * band_fraction
    quantile_band = float(np.quantile(wall_distance, 0.22))
    return float(max(fraction_band, quantile_band))


def circular_smooth(path: np.ndarray, passes: int = 2):
    smoothed = path.copy()
    for _ in range(passes):
        smoothed = (
            np.roll(smoothed, 1, axis=0)
            + 2.0 * smoothed
            + np.roll(smoothed, -1, axis=0)
        ) / 4.0
    return smoothed


def build_centroid_path(embedding: np.ndarray, progress: np.ndarray, mask: np.ndarray, n_bins: int):
    centers = np.linspace(0.0, 1.0, n_bins, endpoint=False) + 0.5 / n_bins
    bin_indices = np.floor(progress * n_bins).astype(int) % n_bins

    path = np.full((n_bins, embedding.shape[1]), np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        take = mask & (bin_indices == idx)
        counts[idx] = int(np.sum(take))
        if counts[idx] > 0:
            path[idx] = embedding[take].mean(axis=0)

    valid = ~np.isnan(path[:, 0])
    if valid.sum() < max(6, n_bins // 5):
        raise ValueError("Not enough perimeter-band samples to build a stable centroid path.")

    valid_idx = np.where(valid)[0]
    for idx in range(n_bins):
        if valid[idx]:
            continue
        right = valid_idx[valid_idx > idx]
        left = valid_idx[valid_idx < idx]
        next_idx = int(right[0]) if right.size else int(valid_idx[0])
        prev_idx = int(left[-1]) if left.size else int(valid_idx[-1])

        forward = (next_idx - idx) % n_bins
        backward = (idx - prev_idx) % n_bins
        total = forward + backward
        if total == 0:
            path[idx] = path[prev_idx]
        else:
            mix = backward / total
            path[idx] = (1.0 - mix) * path[prev_idx] + mix * path[next_idx]

    path = circular_smooth(path, passes=2)
    path_closed = np.vstack([path, path[0]])
    progress_closed = np.append(centers, 1.0)
    return path_closed, progress_closed, counts


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


def set_projection(ax, proj_type: str):
    try:
        ax.set_proj_type(proj_type)
    except Exception:
        pass


def set_view(ax, elev: float, azim: float, roll: float):
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def plot_colored_path(ax, path: np.ndarray, progress: np.ndarray, cmap, line_width: float, alpha: float, linestyle: str = "-"):
    for idx in range(len(path) - 1):
        color = cmap(progress[idx])
        ax.plot(
            path[idx : idx + 2, 0],
            path[idx : idx + 2, 1],
            path[idx : idx + 2, 2],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
        )


def scatter_points(ax, embedding: np.ndarray, progress: np.ndarray, mask: np.ndarray, cmap, alpha: float):
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        embedding[mask, 2],
        c=progress[mask],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=7,
        alpha=alpha,
        edgecolors="none",
        depthshade=False,
    )


def add_dataset_title(ax, text: str):
    ax.set_title(text, color="#f3f3f3", fontsize=12, pad=10)


def build_static_figure(
    run_dir: Path,
    loaded,
    paths,
    centers,
    radius,
    cmap,
    elev: float,
    azim: float,
    roll: float,
    proj_type: str,
):
    fig = plt.figure(figsize=(16, 14), facecolor="#060606")
    panel_order = [
        ("train", "A train"),
        ("holdout", "A held out"),
        ("b", "B"),
        ("overlay", "A vs B perimeter path"),
    ]

    axes = []
    for idx, (key, title) in enumerate(panel_order, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        style_axis(ax, centers, radius)
        set_projection(ax, proj_type)
        set_view(ax, elev=elev, azim=azim, roll=roll)
        add_dataset_title(ax, title)
        axes.append(ax)

        if key == "overlay":
            scatter_points(ax, loaded["train"]["embedding"], loaded["train"]["progress"], loaded["train"]["mask"], cmap, 0.035)
            scatter_points(ax, loaded["b"]["embedding"], loaded["b"]["progress"], loaded["b"]["mask"], cmap, 0.035)
            plot_colored_path(
                ax,
                paths["train"]["path"],
                paths["train"]["path_progress"],
                cmap,
                line_width=3.0,
                alpha=0.45,
                linestyle="--",
            )
            plot_colored_path(
                ax,
                paths["b"]["path"],
                paths["b"]["path_progress"],
                cmap,
                line_width=5.4,
                alpha=0.98,
                linestyle="-",
            )
            ax.text2D(0.03, 0.92, "Dashed = A train\nSolid = B", transform=ax.transAxes, color="#f3f3f3", fontsize=10)
            continue

        scatter_points(ax, loaded[key]["embedding"], loaded[key]["progress"], loaded[key]["mask"], cmap, loaded[key]["point_alpha"])
        plot_colored_path(
            ax,
            paths[key]["path"],
            paths[key]["path_progress"],
            cmap,
            line_width=loaded[key]["line_width"],
            alpha=loaded[key]["path_alpha"],
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
        "Same clockwise perimeter definition in A and B; path color encodes shared perimeter progress.",
        ha="center",
        va="center",
        color="#f3f3f3",
        fontsize=12,
    )

    svg_path = run_dir / "perimeter_progress_static.svg"
    png_path = run_dir / "perimeter_progress_static.png"
    fig.savefig(svg_path, dpi=250, facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return svg_path, png_path


def render_movie(
    run_dir: Path,
    loaded,
    paths,
    centers,
    radius,
    cmap,
    fps: int,
    dpi: int,
    elev: float,
    azim: float,
    roll: float,
    proj_type: str,
):
    video_path = run_dir / "perimeter_progress_paths.mp4"
    preview_path = run_dir / "perimeter_progress_paths_preview.png"

    fig = plt.figure(figsize=(8.8, 8.8), facecolor="#060606")
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius)
    set_projection(ax, proj_type)
    set_view(ax, elev=elev, azim=azim, roll=roll)

    subtitle = fig.text(0.5, 0.06, "", ha="center", va="center", color="#f3f3f3", fontsize=14)

    scatters = {}
    for name in DATASETS:
        scatters[name] = ax.scatter(
            loaded[name]["embedding"][loaded[name]["mask"], 0],
            loaded[name]["embedding"][loaded[name]["mask"], 1],
            loaded[name]["embedding"][loaded[name]["mask"], 2],
            c=loaded[name]["progress"][loaded[name]["mask"]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=9,
            alpha=0.0,
            edgecolors="none",
            depthshade=False,
        )

    line_artists = {name: [] for name in DATASETS}
    endpoint_artists = {name: [] for name in DATASETS}
    for name in DATASETS:
        for idx in range(len(paths[name]["path"]) - 1):
            color = cmap(paths[name]["path_progress"][idx])
            (line,) = ax.plot([], [], [], color=color, linewidth=loaded[name]["line_width"], alpha=0.0, solid_capstyle="round")
            endpoint = ax.scatter([], [], [], s=28, color=[color], alpha=0.0, depthshade=False, edgecolors="none")
            line_artists[name].append(line)
            endpoint_artists[name].append(endpoint)

    title = ax.set_title("", color="#f3f3f3", fontsize=15, pad=10)

    intro = 18
    draw_frames = 96
    hold = 18
    final_hold = 40
    windows = {
        "train": (intro, intro + draw_frames),
        "holdout": (intro + draw_frames + hold, intro + 2 * draw_frames + hold),
        "b": (intro + 2 * draw_frames + 2 * hold, intro + 3 * draw_frames + 2 * hold),
    }
    total_frames = windows["b"][1] + final_hold

    writer = FFMpegWriter(fps=fps, metadata={"title": "Perimeter progress paths"}, bitrate=9000)
    with writer.saving(fig, str(video_path), dpi):
        for frame_idx in range(total_frames):
            if frame_idx < windows["train"][0]:
                title.set_text("Shared physical perimeter variable")
                subtitle.set_text("Same clockwise perimeter definition in A and B")
            elif frame_idx < windows["holdout"][0]:
                title.set_text("Environment A train")
                subtitle.set_text("Perimeter path revealed through latent space")
            elif frame_idx < windows["b"][0]:
                title.set_text("Environment A held out")
                subtitle.set_text("Same perimeter progression, same context")
            elif frame_idx < windows["b"][1]:
                title.set_text("Environment B")
                subtitle.set_text("Same perimeter progression, different latent path")
            else:
                title.set_text("A vs B perimeter path comparison")
                subtitle.set_text("Same physical route variable, different latent geometry across contexts")

            for name in DATASETS:
                start, end = windows[name]
                if frame_idx < start:
                    progress = 0.0
                elif frame_idx >= end:
                    progress = 1.0
                else:
                    progress = (frame_idx - start + 1) / max(1, end - start)

                if name == "train":
                    settled_alpha = 0.14 if frame_idx >= windows["b"][0] else 0.18
                elif name == "holdout":
                    settled_alpha = 0.10 if frame_idx >= windows["b"][0] else 0.14
                else:
                    settled_alpha = 0.14

                if progress == 0.0:
                    scatters[name].set_alpha(0.0)
                else:
                    scatters[name].set_alpha(settled_alpha)

                visible_segments = progress * (len(paths[name]["path"]) - 1)
                for seg_idx, line in enumerate(line_artists[name]):
                    segment_progress = visible_segments - seg_idx
                    if frame_idx >= windows["b"][1] and name in {"train", "holdout"}:
                        line_alpha = 0.24 if name == "train" else 0.14
                    elif progress > 0.0:
                        line_alpha = loaded[name]["path_alpha"]
                    else:
                        line_alpha = 0.0

                    line.set_alpha(line_alpha)
                    endpoint_artists[name][seg_idx].set_alpha(line_alpha)

                    if segment_progress <= 0:
                        line.set_data([], [])
                        line.set_3d_properties([])
                        endpoint_artists[name][seg_idx]._offsets3d = ([], [], [])
                        continue

                    start_point = paths[name]["path"][seg_idx]
                    end_point = paths[name]["path"][seg_idx + 1]
                    if segment_progress >= 1:
                        current_end = end_point
                    else:
                        current_end = start_point + segment_progress * (end_point - start_point)

                    line.set_data(
                        [start_point[0], current_end[0]],
                        [start_point[1], current_end[1]],
                    )
                    line.set_3d_properties([start_point[2], current_end[2]])
                    endpoint_artists[name][seg_idx]._offsets3d = (
                        [current_end[0]],
                        [current_end[1]],
                        [current_end[2]],
                    )

            if frame_idx == windows["b"][1]:
                fig.savefig(preview_path, dpi=dpi, facecolor=fig.get_facecolor())

            writer.grab_frame()

    plt.close(fig)
    return video_path, preview_path, windows, total_frames


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    loaded = load_data(run_dir)

    bounds = robust_arena_bounds(loaded)
    band_width = compute_perimeter_band(loaded, bounds, args.band_fraction)
    cmap = colormaps.get_cmap("turbo_r")

    paths = {}
    for name in DATASETS:
        progress, wall_distance = perimeter_progress(loaded[name]["position"], bounds)
        mask = wall_distance <= band_width
        loaded[name]["progress"] = progress
        loaded[name]["wall_distance"] = wall_distance
        loaded[name]["mask"] = mask

        path, path_progress, counts = build_centroid_path(
            loaded[name]["embedding"],
            progress,
            mask,
            args.bins,
        )
        paths[name] = {
            "path": path,
            "path_progress": path_progress,
            "counts": counts,
        }

    centers, radius = compute_limits(loaded)
    static_svg, static_png = build_static_figure(
        run_dir,
        loaded,
        paths,
        centers,
        radius,
        cmap,
        elev=args.elev,
        azim=args.azim,
        roll=args.roll,
        proj_type=args.proj_type,
    )
    video_path, preview_path, windows, total_frames = render_movie(
        run_dir,
        loaded,
        paths,
        centers,
        radius,
        cmap,
        fps=args.fps,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        roll=args.roll,
        proj_type=args.proj_type,
    )

    np.savez_compressed(
        run_dir / "perimeter_progress_artifacts.npz",
        train_progress=loaded["train"]["progress"],
        train_mask=loaded["train"]["mask"],
        train_path=paths["train"]["path"],
        holdout_progress=loaded["holdout"]["progress"],
        holdout_mask=loaded["holdout"]["mask"],
        holdout_path=paths["holdout"]["path"],
        b_progress=loaded["b"]["progress"],
        b_mask=loaded["b"]["mask"],
        b_path=paths["b"]["path"],
    )

    metadata = {
        "run_dir": str(run_dir),
        "arena_bounds": bounds,
        "band_width": band_width,
        "n_bins": args.bins,
        "fps": args.fps,
        "dpi": args.dpi,
        "elev": args.elev,
        "azim": args.azim,
        "roll": args.roll,
        "proj_type": args.proj_type,
        "dataset_stats": {
            name: {
                "display": loaded[name]["display"],
                "n_points": int(loaded[name]["embedding"].shape[0]),
                "n_perimeter_points": int(np.sum(loaded[name]["mask"])),
                "min_bin_count": int(paths[name]["counts"].min()),
                "max_bin_count": int(paths[name]["counts"].max()),
            }
            for name in DATASETS
        },
        "artifacts": {
            "static_svg": str(static_svg),
            "static_png": str(static_png),
            "video_mp4": str(video_path),
            "video_preview_png": str(preview_path),
            "perimeter_npz": str(run_dir / "perimeter_progress_artifacts.npz"),
        },
        "movie_windows": windows,
        "total_frames": total_frames,
        "notes": {
            "perimeter_definition": "Shared clockwise rectangular perimeter beginning at the lower-left corner of pooled arena bounds.",
            "band_definition": "Samples within a shared perimeter band are used for the centroid path and recolored raw points.",
            "scientific_intent": "Same physical perimeter progression is mapped into latent space to compare context-specific geometry in A vs B.",
        },
    }
    with open(run_dir / "perimeter_progress_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved static figure to {static_svg}")
    print(f"Saved movie to {video_path}")


if __name__ == "__main__":
    main()
