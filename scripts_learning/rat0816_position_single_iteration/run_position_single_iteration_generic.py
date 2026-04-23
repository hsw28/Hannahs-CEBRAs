import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import cebra
from cebra import CEBRA

from ca_velocity import ca_velocity
from smoothpos import smoothpos


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single position comparison and save a 3-panel visualization.")
    parser.add_argument("traceA1An_An", type=str)
    parser.add_argument("traceAnB1_An", type=str)
    parser.add_argument("traceA1An_A1", type=str)
    parser.add_argument("traceAnB1_B1", type=str)
    parser.add_argument("PosAn", type=str)
    parser.add_argument("PosA1", type=str)
    parser.add_argument("PosB1", type=str)
    parser.add_argument("--run-label", type=str, default="position_single")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--temp-mode", type=str, default="auto", choices=["auto", "constant"])
    parser.add_argument("--min-temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trace(path: str) -> np.ndarray:
    return np.transpose(cebra.load_data(file=path))


def load_position(path: str) -> np.ndarray:
    return cebra.load_data(file=path)


def preprocess_position(position: np.ndarray, trace_length: int):
    position = smoothpos(position)
    position = position[:, 1:]
    position = position[::2]
    if len(position) > trace_length:
        position = position[:trace_length]
    velocity = ca_velocity(position)
    high_velocity_indices = np.where(velocity >= 4)[0]
    if high_velocity_indices.size > 0 and high_velocity_indices[-1] + 1 < len(position):
        high_velocity_indices = high_velocity_indices + 1
    return position[high_velocity_indices], high_velocity_indices


def hold_out_middle_chunk(data: np.ndarray, percent_to_train: float):
    if percent_to_train > 1:
        percent_to_train = percent_to_train / 100.0

    total_length = len(data)
    train_fraction = float(percent_to_train)
    hold_fraction = max(0.0, 1.0 - train_fraction)
    hold_length = int(total_length * hold_fraction)

    if hold_length <= 0:
        return np.array(data), np.array(data[:0])

    start_hold = int((total_length - hold_length) / 2)
    end_hold = start_hold + hold_length

    train = np.concatenate((data[:start_hold], data[end_hold:]), axis=0)
    holdout = data[start_hold:end_hold]
    return np.array(train), np.array(holdout)


def create_model(args):
    common_kwargs = dict(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=args.learning_rate,
        output_dimension=3,
        max_iterations=args.max_iterations,
        distance=args.distance,
        conditional="time_delta",
        device="cuda_if_available",
        num_hidden_units=32,
        time_offsets=1,
        verbose=True,
    )
    if args.temp_mode == "auto":
        common_kwargs["temperature_mode"] = "auto"
        if args.min_temperature is not None:
            common_kwargs["min_temperature"] = args.min_temperature
        return CEBRA(**common_kwargs)

    common_kwargs["temperature_mode"] = "constant"
    if args.min_temperature is None:
        raise ValueError("--min-temperature is required when --temp-mode constant")
    common_kwargs["temperature"] = args.min_temperature
    return CEBRA(**common_kwargs)


def normalized_corner_distance(position: np.ndarray) -> np.ndarray:
    min_x_index = np.argmin(position[:, 0])
    corner = np.array([np.min(position[:, 0]), position[min_x_index, 1]])
    distances = np.linalg.norm(position - corner, axis=1)
    max_distance = np.max(distances)
    if max_distance == 0:
        return np.zeros_like(distances)
    return distances / max_distance


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("latent1", labelpad=-8)
    ax.set_ylabel("latent2", labelpad=-8)
    ax.set_zlabel("latent3", labelpad=-8)
    ax.grid(False)


def build_figure(artifacts, figure_path: Path, preview_path: Path) -> None:
    fig = plt.figure(figsize=(16, 5.5))
    titles = [
        ("train", "Model from A(n) training data"),
        ("holdout", "Model applied to A(n) held out data"),
        ("b", "Model applied to B(1)"),
    ]

    scatter = None
    for idx, (key, title) in enumerate(titles, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        values = artifacts[key]["distance"]
        embedding = artifacts[key]["embedding"]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=values,
            cmap="rainbow",
            s=7,
            alpha=0.9,
            edgecolors="none",
            vmin=0.0,
            vmax=1.0,
        )
        style_axis(ax, title)

    cbar = fig.colorbar(scatter, ax=fig.axes, fraction=0.025, pad=0.04)
    cbar.set_label("Normalized distance from corner of envl. (a.u.)", rotation=270, labelpad=18)
    fig.tight_layout()
    fig.savefig(figure_path, format=figure_path.suffix.lstrip("."), dpi=250)
    fig.savefig(preview_path, format=preview_path.suffix.lstrip("."), dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{args.run_label}_{timestamp}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_a1an_an = load_trace(args.traceA1An_An)
    trace_anb1_an = load_trace(args.traceAnB1_An)
    trace_anb1_b1 = load_trace(args.traceAnB1_B1)

    pos_an_raw = load_position(args.PosAn)
    pos_b1_raw = load_position(args.PosB1)

    pos_an, idx_an = preprocess_position(pos_an_raw, len(trace_a1an_an))
    pos_b1, idx_b1 = preprocess_position(pos_b1_raw, len(trace_anb1_b1))

    trace_a1an_an = trace_a1an_an[idx_an]
    trace_anb1_an = trace_anb1_an[idx_an]
    trace_anb1_b1 = trace_anb1_b1[idx_b1]

    trace_train, trace_holdout = hold_out_middle_chunk(trace_anb1_an, 75)
    pos_train, pos_holdout = hold_out_middle_chunk(pos_an, 75)

    model = create_model(args)
    model.fit(trace_train, pos_train)

    emb_train = model.transform(trace_train)
    emb_holdout = model.transform(trace_holdout)
    emb_b = model.transform(trace_anb1_b1)

    artifacts = {
        "train": {"embedding": emb_train, "distance": normalized_corner_distance(pos_train), "position": pos_train},
        "holdout": {"embedding": emb_holdout, "distance": normalized_corner_distance(pos_holdout), "position": pos_holdout},
        "b": {"embedding": emb_b, "distance": normalized_corner_distance(pos_b1), "position": pos_b1},
    }

    figure_path = run_dir / "position_embeddings.svg"
    preview_path = run_dir / "position_embeddings_preview.png"
    build_figure(artifacts, figure_path, preview_path)

    np.savez_compressed(
        run_dir / "embeddings_and_labels.npz",
        emb_train=emb_train,
        dist_train=artifacts["train"]["distance"],
        pos_train=pos_train,
        emb_holdout=emb_holdout,
        dist_holdout=artifacts["holdout"]["distance"],
        pos_holdout=pos_holdout,
        emb_b=emb_b,
        dist_b=artifacts["b"]["distance"],
        pos_b=pos_b1,
    )

    model_path = run_dir / "model_b1_matched.pt"
    model.save(str(model_path))

    metadata = {
        "timestamp": timestamp,
        "run_label": args.run_label,
        "seed": args.seed,
        "parameters": {
            "learning_rate": args.learning_rate,
            "max_iterations": args.max_iterations,
            "distance": args.distance,
            "temp_mode": args.temp_mode,
            "min_temperature": args.min_temperature,
            "output_dimension": 3,
            "holdout_strategy": "middle_25_percent_chunk",
        },
        "source_files": {
            "traceA1An_An": args.traceA1An_An,
            "traceAnB1_An": args.traceAnB1_An,
            "traceA1An_A1": args.traceA1An_A1,
            "traceAnB1_B1": args.traceAnB1_B1,
            "PosAn": args.PosAn,
            "PosA1": args.PosA1,
            "PosB1": args.PosB1,
        },
        "artifacts": {
            "embedding_plot": str(figure_path),
            "preview_png": str(preview_path),
            "embeddings_npz": str(run_dir / "embeddings_and_labels.npz"),
            "model_b1_matched": str(model_path),
        },
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
