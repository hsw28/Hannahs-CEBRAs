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


def parse_args():
    parser = argparse.ArgumentParser(description="Render a rotating 3D epoch-trajectory video from saved embedding artifacts.")
    parser.add_argument("--run-dir", required=True, type=str, help="Path to a single run directory containing embeddings_and_labels.npz.")
    parser.add_argument("--dataset-key", default="external", choices=["external", "train", "holdout"], help="Which embedding set to visualize.")
    parser.add_argument("--fps", default=18, type=int)
    parser.add_argument("--frames", default=240, type=int)
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--elev", default=22.0, type=float)
    parser.add_argument("--reveal-fraction", default=0.3, type=float, help="Fraction of frames spent revealing trajectories before rotation continues.")
    return parser.parse_args()


def load_dataset(npz_path: Path, dataset_key: str):
    data = np.load(npz_path)
    key_map = {
        "train": ("emb_b_train", "labels_b_train"),
        "holdout": ("emb_b_holdout", "labels_b_holdout"),
        "external": ("emb_b_external", "labels_b1"),
    }
    emb_key, label_key = key_map[dataset_key]
    return data[emb_key], data[label_key]


def split_into_epochs(embedding: np.ndarray, labels: np.ndarray):
    unique_labels = np.unique(labels)
    epoch_length = len(unique_labels)
    if epoch_length <= 1:
        raise ValueError("Need more than one label/bin to infer trajectories.")
    if len(labels) % epoch_length != 0:
        raise ValueError(f"Label count {len(labels)} is not divisible by inferred epoch length {epoch_length}.")

    expected = np.arange(1, epoch_length + 1)
    epochs = []
    for start in range(0, len(labels), epoch_length):
        epoch_labels = labels[start : start + epoch_length]
        epoch_embedding = embedding[start : start + epoch_length]
        if not np.array_equal(epoch_labels, expected):
            raise ValueError(
                f"Epoch starting at index {start} does not match expected label sequence {expected.tolist()}; got {epoch_labels.tolist()}."
            )
        epochs.append(epoch_embedding)
    return np.asarray(epochs), expected


def compute_limits(epochs: np.ndarray):
    mins = epochs.reshape(-1, 3).min(axis=0)
    maxs = epochs.reshape(-1, 3).max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius *= 1.12
    return centers, radius


def style_axis(ax, centers: np.ndarray, radius: float):
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


def make_colors(n_epochs: int):
    cmap = matplotlib.colormaps.get_cmap("turbo").resampled(n_epochs)
    return [cmap(i) for i in range(n_epochs)]


def render_video(run_dir: Path, dataset_key: str, fps: int, frames: int, dpi: int, elev: float, reveal_fraction: float):
    npz_path = run_dir / "embeddings_and_labels.npz"
    embedding, labels = load_dataset(npz_path, dataset_key)
    epochs, expected_labels = split_into_epochs(embedding, labels)
    n_epochs, epoch_length, _ = epochs.shape

    output_name = f"{run_dir.name}_{dataset_key}_epoch_trajectories.mp4"
    output_path = run_dir / output_name
    preview_path = run_dir / f"{run_dir.name}_{dataset_key}_epoch_trajectories_preview.png"

    colors = make_colors(n_epochs)
    centers, radius = compute_limits(epochs)

    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_axis(ax, centers, radius)

    ax.set_title(f"{run_dir.name} | {dataset_key} | {n_epochs} epochs x {epoch_length} bins", pad=16)

    line_artists = []
    point_artists = []
    for color in colors:
        (line,) = ax.plot([], [], [], lw=1.7, color=color, alpha=0.95)
        points = ax.scatter([], [], [], s=18, color=[color], alpha=0.95, depthshade=False)
        line_artists.append(line)
        point_artists.append(points)

    writer = FFMpegWriter(fps=fps, metadata={"title": output_name}, bitrate=8000)
    reveal_frames = max(1, int(frames * reveal_fraction))
    azims = np.linspace(35, 395, frames)

    with writer.saving(fig, str(output_path), dpi):
        for frame_idx in range(frames):
            reveal_progress = min(1.0, (frame_idx + 1) / reveal_frames)
            visible_points = max(1, int(np.ceil(reveal_progress * epoch_length)))

            for epoch_idx, epoch in enumerate(epochs):
                current = epoch[:visible_points]
                line_artists[epoch_idx].set_data(current[:, 0], current[:, 1])
                line_artists[epoch_idx].set_3d_properties(current[:, 2])
                point_artists[epoch_idx]._offsets3d = (current[:, 0], current[:, 1], current[:, 2])

            ax.view_init(elev=elev, azim=float(azims[frame_idx]))

            if frame_idx == 0:
                fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")

            writer.grab_frame()

    plt.close(fig)

    metadata = {
        "run_dir": str(run_dir),
        "dataset_key": dataset_key,
        "epochs": int(n_epochs),
        "bins_per_epoch": int(epoch_length),
        "fps": fps,
        "frames": frames,
        "dpi": dpi,
        "elev": elev,
        "reveal_fraction": reveal_fraction,
        "expected_labels": expected_labels.tolist(),
        "video_path": str(output_path),
        "preview_path": str(preview_path),
    }
    with open(run_dir / f"{run_dir.name}_{dataset_key}_epoch_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved video to {output_path}")
    print(f"Saved preview to {preview_path}")


def main():
    args = parse_args()
    render_video(
        run_dir=Path(args.run_dir).resolve(),
        dataset_key=args.dataset_key,
        fps=args.fps,
        frames=args.frames,
        dpi=args.dpi,
        elev=args.elev,
        reveal_fraction=args.reveal_fraction,
    )


if __name__ == "__main__":
    main()
