import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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
from hold_out import hold_out
from smoothpos import smoothpos


PARAMETER_SET = {
    "rat5_position": {
        "learning_rate": 1.0e-3,
        "min_temperature": None,
        "max_iterations": 18000,
        "distance": "cosine",
        "temp_mode": "auto",
        "output_dimension": 3,
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single Rat 5 position comparison and save a 3-panel visualization."
    )
    parser.add_argument("traceA1An_An", type=str)
    parser.add_argument("traceAnB1_An", type=str)
    parser.add_argument("traceA1An_A1", type=str)
    parser.add_argument("traceAnB1_B1", type=str)
    parser.add_argument("PosAn", type=str)
    parser.add_argument("PosA1", type=str)
    parser.add_argument("PosB1", type=str)
    parser.add_argument("--parameter_set_name", type=str, default="rat5_position")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "rat0816_single_iteration_pos"),
    )
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


def preprocess_position(position: np.ndarray, trace_length: int) -> np.ndarray:
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


def apply_indices(trace: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return trace[indices]


def create_model(parameter_set: Dict[str, object]) -> CEBRA:
    common_kwargs = dict(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=parameter_set["learning_rate"],
        temperature_mode=parameter_set["temp_mode"],
        output_dimension=parameter_set["output_dimension"],
        max_iterations=parameter_set["max_iterations"],
        distance=parameter_set["distance"],
        conditional="time_delta",
        device="cuda_if_available",
        num_hidden_units=32,
        time_offsets=1,
        verbose=True,
    )
    if parameter_set["min_temperature"] is not None:
        common_kwargs["min_temperature"] = parameter_set["min_temperature"]
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


def build_figure(artifacts: Dict[str, Dict[str, np.ndarray]], figure_path: Path, preview_path: Path) -> None:
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

    if args.parameter_set_name not in PARAMETER_SET:
        raise KeyError(
            f"Unknown parameter_set_name '{args.parameter_set_name}'. Available: {sorted(PARAMETER_SET)}"
        )

    parameter_set = PARAMETER_SET[args.parameter_set_name]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{args.parameter_set_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_a1an_an = load_trace(args.traceA1An_An)
    trace_anb1_an = load_trace(args.traceAnB1_An)
    trace_anb1_b1 = load_trace(args.traceAnB1_B1)

    pos_an_raw = load_position(args.PosAn)
    pos_b1_raw = load_position(args.PosB1)

    pos_an, idx_an = preprocess_position(pos_an_raw, len(trace_a1an_an))
    pos_b1, idx_b1 = preprocess_position(pos_b1_raw, len(trace_anb1_b1))

    trace_a1an_an = apply_indices(trace_a1an_an, idx_an)
    trace_anb1_an = apply_indices(trace_anb1_an, idx_an)
    trace_anb1_b1 = apply_indices(trace_anb1_b1, idx_b1)

    trace_train, trace_holdout = hold_out(trace_anb1_an, 75)
    pos_train, pos_holdout = hold_out(pos_an, 75)

    model = create_model(parameter_set)
    model.fit(trace_train, pos_train)

    emb_train = model.transform(trace_train)
    emb_holdout = model.transform(trace_holdout)
    emb_b = model.transform(trace_anb1_b1)

    artifacts = {
        "train": {"embedding": emb_train, "distance": normalized_corner_distance(pos_train), "position": pos_train},
        "holdout": {
            "embedding": emb_holdout,
            "distance": normalized_corner_distance(pos_holdout),
            "position": pos_holdout,
        },
        "b": {"embedding": emb_b, "distance": normalized_corner_distance(pos_b1), "position": pos_b1},
    }

    figure_path = run_dir / "rat0816_position_embeddings.svg"
    preview_path = run_dir / "rat0816_position_embeddings_preview.png"
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
        "parameter_set_name": args.parameter_set_name,
        "parameter_set": parameter_set,
        "seed": args.seed,
        "source_files": {
            "traceA1An_An": args.traceA1An_An,
            "traceAnB1_An": args.traceAnB1_An,
            "traceA1An_A1": args.traceA1An_A1,
            "traceAnB1_B1": args.traceAnB1_B1,
            "PosAn": args.PosAn,
            "PosA1": args.PosA1,
            "PosB1": args.PosB1,
        },
        "notes": {
            "figure_panels": ["training_75", "held_out_25", "B1"],
            "label_definition": "Normalized distance from a detected environment corner in arbitrary units.",
            "preprocessing": "Smooth positions, drop time column, downsample by 2, keep high-velocity frames (>=4), then split 75/25.",
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
