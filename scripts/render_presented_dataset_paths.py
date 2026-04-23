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
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

RNG = np.random.default_rng(81610)


DATASETS = {
    "train": {
        "embedding_key": "emb_b_train",
        "label_key": "labels_b_train",
        "display": "Training A(n)",
        "saturation_scale": 1.0,
        "value_scale": 0.95,
        "line_width": 4.0,
        "point_alpha_active": 0.34,
        "point_alpha_settled": 0.22,
    },
    "holdout": {
        "embedding_key": "emb_b_holdout",
        "label_key": "labels_b_holdout",
        "display": "Held-out Test A(n)",
        "saturation_scale": 0.48,
        "value_scale": 0.8,
        "line_width": 3.6,
        "point_alpha_active": 0.22,
        "point_alpha_settled": 0.14,
    },
    "external": {
        "embedding_key": "emb_b_external",
        "label_key": "labels_b1",
        "display": "New Environment B1",
        "saturation_scale": 1.28,
        "value_scale": 1.14,
        "line_width": 5.6,
        "point_alpha_active": 0.48,
        "point_alpha_settled": 0.34,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render a presentation-style 3D video with epoch-colored points and average paths drawn in sequence.")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--fps", default=18, type=int)
    parser.add_argument("--frames", default=270, type=int)
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--elev", default=22.0, type=float)
    parser.add_argument("--base-azim", default=312.5, type=float)
    parser.add_argument("--output-suffix", default="", type=str)
    parser.add_argument("--hide-points", action="store_true")
    parser.add_argument("--dark-background", action="store_true")
    return parser.parse_args()


def load_data(npz_path: Path):
    npz = np.load(npz_path)
    loaded = {}
    for name, spec in DATASETS.items():
        loaded[name] = {
            "embedding": npz[spec["embedding_key"]],
            "labels": npz[spec["label_key"]],
            "display": spec["display"],
            "saturation_scale": spec["saturation_scale"],
            "value_scale": spec["value_scale"],
            "line_width": spec["line_width"],
            "point_alpha_active": spec["point_alpha_active"],
            "point_alpha_settled": spec["point_alpha_settled"],
        }
    return loaded


def infer_epoch_ids(labels: np.ndarray) -> np.ndarray:
    epoch_ids = np.arange(len(labels)) // 10
    return epoch_ids.astype(int)


def compute_average_path(embedding: np.ndarray, labels: np.ndarray):
    bins = sorted(int(v) for v in np.unique(labels))
    avg = []
    for bin_value in bins:
        mask = labels == bin_value
        avg.append(embedding[mask].mean(axis=0))
    return np.asarray(avg), bins


def compute_limits(loaded):
    stacked = np.concatenate([item["embedding"] for item in loaded.values()], axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.12
    return centers, radius


def jitter_points(points: np.ndarray, radius: float) -> np.ndarray:
    jitter_scale = radius * 0.018
    noise = RNG.normal(loc=0.0, scale=jitter_scale, size=points.shape)
    return points + noise


def style_axis(ax, centers, radius, dark_background: bool):
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
    ax.set_facecolor("#0a0a0a" if dark_background else "white")
    ax.grid(False)


def make_epoch_palette(max_epochs: int):
    cmap = colormaps.get_cmap("turbo")
    if max_epochs <= 1:
        return [cmap(0.0)]
    return [cmap(v) for v in np.linspace(0.04, 0.96, max_epochs)]


def make_segment_palette(n_segments: int):
    cmap = colormaps.get_cmap("turbo")
    if n_segments <= 1:
        return [cmap(0.12)]
    return [cmap(v) for v in np.linspace(0.08, 0.92, n_segments)]


def adjust_color(color, saturation_scale: float, value_scale: float):
    rgba = np.array(mcolors.to_rgba(color))
    hsv = mcolors.rgb_to_hsv(rgba[:3])
    hsv[1] = np.clip(hsv[1] * saturation_scale, 0.0, 1.0)
    hsv[2] = np.clip(hsv[2] * value_scale, 0.0, 1.0)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb[0], rgb[1], rgb[2], rgba[3])


def blend_colors(color_a, color_b, mix: float):
    a = np.array(mcolors.to_rgba(color_a))
    b = np.array(mcolors.to_rgba(color_b))
    mixed = (1.0 - mix) * a + mix * b
    return tuple(mixed.tolist())


def dataset_alpha(frame_idx: int, dataset_name: str, line_windows):
    start, end = line_windows[dataset_name]
    if frame_idx < start:
        return 0.0
    if dataset_name == "external" and frame_idx >= start:
        return 1.0
    if frame_idx <= end:
        return 1.0
    if frame_idx >= line_windows["external"][0]:
        return 0.12
    return 0.35


def presentation_text(frame_idx: int, line_windows):
    if frame_idx < line_windows["train"][0]:
        return "Epoch-colored points"
    if frame_idx < line_windows["holdout"][0]:
        return "Training path"
    if frame_idx < line_windows["external"][0]:
        return "Training path, then held-out test path"
    return "Training path, held-out test path, and new environment path"


def render_video(run_dir: Path, fps: int, frames: int, dpi: int, elev: float, base_azim: float, output_suffix: str, hide_points: bool, dark_background: bool):
    npz_path = run_dir / "embeddings_and_labels.npz"
    loaded = load_data(npz_path)
    avg_paths = {name: compute_average_path(item["embedding"], item["labels"])[0] for name, item in loaded.items()}
    centers, radius = compute_limits(loaded)

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_base = run_dir.name + f"_presented_dataset_paths{suffix}"
    video_path = run_dir / f"{output_base}.mp4"
    preview_path = run_dir / f"{output_base}_preview.png"
    metadata_path = run_dir / f"{output_base}.json"

    max_epochs = max(int(infer_epoch_ids(item["labels"]).max()) + 1 for item in loaded.values())
    n_segments = max(len(path) - 1 for path in avg_paths.values())
    palette = make_epoch_palette(max_epochs)
    segment_palette = make_segment_palette(n_segments)

    background = "#0a0a0a" if dark_background else "white"
    foreground = "#f2f2f2" if dark_background else "#111111"
    fig = plt.figure(figsize=(8.5, 8.5), facecolor=background)
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius, dark_background=dark_background)

    point_artists = {}
    point_colors = {}
    for name, item in loaded.items():
        emb = item["embedding"]
        bin_labels = sorted(int(v) for v in np.unique(item["labels"]))
        artists = []
        colors = []
        for bin_idx, bin_label in enumerate(bin_labels):
            mask = item["labels"] == bin_label
            base_color = segment_palette[bin_idx % len(segment_palette)]
            color = adjust_color(
                base_color,
                item["saturation_scale"],
                item["value_scale"],
            )
            jittered = jitter_points(emb[mask], radius)
            points = ax.scatter(
                jittered[:, 0],
                jittered[:, 1],
                jittered[:, 2],
                s=34,
                color=color,
                alpha=0.0,
                depthshade=False,
                edgecolors="none",
            )
            artists.append(points)
            colors.append(color)
        point_artists[name] = artists
        point_colors[name] = colors

    line_artists = {}
    endpoint_artists = {}
    line_colors = {}
    for name, item in loaded.items():
        segments = []
        endpoints = []
        colors = []
        dataset_segments = len(avg_paths[name]) - 1
        for seg_idx in range(dataset_segments):
            base_color = segment_palette[seg_idx % len(segment_palette)]
            color = adjust_color(
                base_color,
                item["saturation_scale"],
                item["value_scale"],
            )
            (line,) = ax.plot([], [], [], color=color, linewidth=item["line_width"], alpha=0.98, linestyle="-")
            endpoint = ax.scatter([], [], [], s=42, color=color, alpha=1.0, depthshade=False, edgecolors="none")
            segments.append(line)
            endpoints.append(endpoint)
            colors.append(color)
        line_artists[name] = segments
        endpoint_artists[name] = endpoints
        line_colors[name] = colors

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=adjust_color("#d0d0d0" if dark_background else "#888888", item["saturation_scale"], item["value_scale"]),
            label=item["display"],
            lw=item["line_width"],
            linestyle="-",
        )
        for item in loaded.values()
    ]
    legend = ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=10, handlelength=1.8)
    for text in legend.get_texts():
        text.set_color(foreground)

    title = ax.set_title(run_dir.name, pad=16, color=foreground)
    subtitle = fig.text(0.5, 0.06, "", ha="center", va="center", fontsize=13, color=foreground)

    intro_frames = 18
    segment_frames = 70
    pause_frames = 14
    line_windows = {
        "train": (intro_frames, intro_frames + segment_frames),
        "holdout": (intro_frames + segment_frames + pause_frames, intro_frames + 2 * segment_frames + pause_frames),
        "external": (intro_frames + 2 * segment_frames + 2 * pause_frames, intro_frames + 3 * segment_frames + 2 * pause_frames),
    }
    final_compare_pause_frames = 18
    orbit_start = line_windows["external"][1] + final_compare_pause_frames
    orbit_span = max(1, frames - orbit_start)
    orbit_amplitude = 10.0
    b_focus_start = line_windows["external"][0]
    b_focus_transition_frames = 12
    ghost_gray = mcolors.to_rgba("#d8d8d8" if dark_background else "#d7d7d7")
    faint_gray = mcolors.to_rgba("#4a4a4a" if dark_background else "#ececec")

    writer = FFMpegWriter(fps=fps, metadata={"title": output_base}, bitrate=8000)
    with writer.saving(fig, str(video_path), dpi):
        for idx in range(frames):
            if idx < orbit_start:
                azim = base_azim
            else:
                orbit_progress = min(1.0, (idx - orbit_start) / orbit_span)
                azim = base_azim + orbit_amplitude * orbit_progress
            ax.view_init(elev=elev, azim=float(azim))
            subtitle.set_text(presentation_text(idx, line_windows))

            for name, item in loaded.items():
                path = avg_paths[name]
                start, end = line_windows[name]
                current_alpha = dataset_alpha(idx, name, line_windows)
                dataset_point_artists = point_artists[name]
                if idx < b_focus_start:
                    b_focus_mix = 0.0
                else:
                    b_focus_mix = min(1.0, (idx - b_focus_start) / max(1, b_focus_transition_frames))

                if idx < start:
                    visible_progress = 0.0
                    epoch_progress = 0.0
                elif idx >= end:
                    visible_progress = len(path) - 1
                    epoch_progress = 1.0
                else:
                    progress = (idx - start + 1) / max(1, end - start)
                    visible_progress = progress * (len(path) - 1)

                if current_alpha > 0:
                    visible_bins = min(len(dataset_point_artists), max(1, int(np.floor(visible_progress)) + 1))
                else:
                    visible_bins = 0
                if hide_points:
                    settled_point_alpha = 0.0
                elif name == "external" and idx >= start:
                    settled_point_alpha = item["point_alpha_active"]
                elif idx >= line_windows["external"][0]:
                    settled_point_alpha = (1.0 - b_focus_mix) * item["point_alpha_settled"] + b_focus_mix * (0.12 if name == "train" else 0.0)
                elif idx <= end:
                    settled_point_alpha = item["point_alpha_active"]
                else:
                    settled_point_alpha = item["point_alpha_settled"]
                for bin_idx, points in enumerate(dataset_point_artists):
                    if idx >= line_windows["external"][0] and name != "external":
                        target_color = ghost_gray if name == "train" else faint_gray
                        points.set_color(blend_colors(point_colors[name][bin_idx], target_color, b_focus_mix))
                    else:
                        points.set_color(point_colors[name][bin_idx])
                    points.set_alpha(settled_point_alpha if bin_idx < visible_bins else 0.0)

                segments = line_artists[name]
                endpoints = endpoint_artists[name]
                for seg_idx, line in enumerate(segments):
                    segment_progress = visible_progress - seg_idx
                    if name == "external":
                        line_alpha = 1.0 if current_alpha > 0 else 0.0
                        endpoint_alpha = 1.0 if current_alpha > 0 else 0.0
                    elif idx >= line_windows["external"][0]:
                        if name == "train":
                            target_line_alpha = 0.20
                            target_endpoint_alpha = 0.0
                            target_line_width = 1.55
                        else:
                            target_line_alpha = 0.0
                            target_endpoint_alpha = 0.0
                            target_line_width = 1.2
                        line_alpha = ((1.0 - b_focus_mix) * current_alpha + b_focus_mix * target_line_alpha) if current_alpha > 0 else 0.0
                        endpoint_alpha = ((1.0 - b_focus_mix) * current_alpha + b_focus_mix * target_endpoint_alpha) if current_alpha > 0 else 0.0
                        line_width = (1.0 - b_focus_mix) * item["line_width"] + b_focus_mix * target_line_width
                    else:
                        line_alpha = current_alpha
                        endpoint_alpha = current_alpha
                        line_width = item["line_width"]
                    if idx >= line_windows["external"][0] and name != "external":
                        target_color = ghost_gray if name == "train" else faint_gray
                        blended_color = blend_colors(line_colors[name][seg_idx], target_color, b_focus_mix)
                        line.set_color(blended_color)
                        endpoints[seg_idx].set_color(blended_color)
                    else:
                        line.set_color(line_colors[name][seg_idx])
                        endpoints[seg_idx].set_color(line_colors[name][seg_idx])
                    line.set_alpha(line_alpha)
                    line.set_linewidth(line_width)
                    if segment_progress <= 0:
                        line.set_data([], [])
                        line.set_3d_properties([])
                        endpoints[seg_idx]._offsets3d = ([], [], [])
                        endpoints[seg_idx].set_alpha(endpoint_alpha)
                        continue

                    start_point = path[seg_idx]
                    end_point = path[seg_idx + 1]
                    if segment_progress >= 1:
                        current_end = end_point
                    else:
                        current_end = start_point + segment_progress * (end_point - start_point)

                    segment_points = np.vstack([start_point, current_end])
                    line.set_data(segment_points[:, 0], segment_points[:, 1])
                    line.set_3d_properties(segment_points[:, 2])

                    if segment_progress > 0:
                        endpoints[seg_idx]._offsets3d = (
                            [current_end[0]],
                            [current_end[1]],
                            [current_end[2]],
                        )
                        endpoints[seg_idx].set_alpha(endpoint_alpha)
                    else:
                        endpoints[seg_idx]._offsets3d = ([], [], [])
                        endpoints[seg_idx].set_alpha(endpoint_alpha)

            if idx == 0:
                fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")

            writer.grab_frame()

    plt.close(fig)

    metadata = {
        "run_dir": str(run_dir),
        "video_path": str(video_path),
        "preview_path": str(preview_path),
        "fps": fps,
        "frames": frames,
        "dpi": dpi,
        "elev": elev,
        "base_azim": base_azim,
        "orbit_start": orbit_start,
        "orbit_amplitude": orbit_amplitude,
        "final_compare_pause_frames": final_compare_pause_frames,
        "line_windows": line_windows,
        "hide_points": hide_points,
        "dark_background": dark_background,
        "b_focus_transition_frames": b_focus_transition_frames,
        "datasets": {
            name: {
                "display": item["display"],
                "saturation_scale": item["saturation_scale"],
                "value_scale": item["value_scale"],
                "line_width": item["line_width"],
                "n_points": int(item["embedding"].shape[0]),
                "n_avg_bins": int(avg_paths[name].shape[0]),
            }
            for name, item in loaded.items()
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved video to {video_path}")
    print(f"Saved preview to {preview_path}")


def main():
    args = parse_args()
    render_video(
        run_dir=Path(args.run_dir).resolve(),
        fps=args.fps,
        frames=args.frames,
        dpi=args.dpi,
        elev=args.elev,
        base_azim=args.base_azim,
        output_suffix=args.output_suffix,
        hide_points=args.hide_points,
        dark_background=args.dark_background,
    )


if __name__ == "__main__":
    main()
