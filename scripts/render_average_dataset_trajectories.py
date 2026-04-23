import argparse
import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D


DATASETS = {
    "train": {
        "embedding_key": "emb_b_train",
        "label_key": "labels_b_train",
        "color": "#2c7fb8",
        "label": "Train A(n)",
    },
    "holdout": {
        "embedding_key": "emb_b_holdout",
        "label_key": "labels_b_holdout",
        "color": "#f28e2b",
        "label": "Held-out A(n)",
    },
    "external": {
        "embedding_key": "emb_b_external",
        "label_key": "labels_b1",
        "color": "#2ca25f",
        "label": "B1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render a rotating 3D video with all points and average dataset trajectories.")
    parser.add_argument("--run-dir", required=True, type=str, help="Path to a single run directory containing embeddings_and_labels.npz.")
    parser.add_argument("--fps", default=18, type=int)
    parser.add_argument("--frames", default=240, type=int)
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--elev", default=22.0, type=float)
    return parser.parse_args()


def load_data(npz_path: Path):
    npz = np.load(npz_path)
    loaded = {}
    for name, spec in DATASETS.items():
        loaded[name] = {
            "embedding": npz[spec["embedding_key"]],
            "labels": npz[spec["label_key"]],
            "color": spec["color"],
            "display": spec["label"],
        }
    return loaded


def compute_average_path(embedding: np.ndarray, labels: np.ndarray):
    bins = sorted(int(v) for v in np.unique(labels))
    points = []
    for bin_value in bins:
        mask = labels == bin_value
        if not np.any(mask):
            continue
        points.append(embedding[mask].mean(axis=0))
    return np.asarray(points), bins


def compute_limits(loaded):
    stacked = np.concatenate([item["embedding"] for item in loaded.values()], axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0) * 1.12
    return centers, radius


def style_axis(ax, centers, radius):
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(False)


def build_legend(loaded):
    return [
        Line2D([0], [0], color=item["color"], lw=2.4, marker="o", markersize=6, label=item["display"])
        for item in loaded.values()
    ]


def render_video(run_dir: Path, fps: int, frames: int, dpi: int, elev: float):
    npz_path = run_dir / "embeddings_and_labels.npz"
    loaded = load_data(npz_path)
    avg_paths = {name: compute_average_path(item["embedding"], item["labels"])[0] for name, item in loaded.items()}
    centers, radius = compute_limits(loaded)

    output_base = run_dir.name + "_average_dataset_trajectories"
    video_path = run_dir / f"{output_base}.mp4"
    preview_path = run_dir / f"{output_base}_preview.png"
    metadata_path = run_dir / f"{output_base}.json"

    fig = plt.figure(figsize=(8.5, 8.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius)
    ax.set_title(f"{run_dir.name} | Train / Held-out / B1 Averages", pad=16)

    for item in loaded.values():
        emb = item["embedding"]
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            emb[:, 2],
            s=10,
            color=item["color"],
            alpha=0.18,
            depthshade=False,
            edgecolors="none",
        )

    for name, item in loaded.items():
        path = avg_paths[name]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=item["color"],
            linewidth=3.2,
            alpha=0.98,
        )
        ax.scatter(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            s=28,
            color=item["color"],
            alpha=1.0,
            depthshade=False,
            edgecolors="none",
        )

    ax.legend(handles=build_legend(loaded), loc="upper left", frameon=False)

    azims = np.linspace(35, 395, frames)
    writer = FFMpegWriter(fps=fps, metadata={"title": output_base}, bitrate=8000)
    with writer.saving(fig, str(video_path), dpi):
        for idx, azim in enumerate(azims):
            ax.view_init(elev=elev, azim=float(azim))
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
        "datasets": {
            name: {
                "display": item["display"],
                "color": item["color"],
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
    )


if __name__ == "__main__":
    main()
