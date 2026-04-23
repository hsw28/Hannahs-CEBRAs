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
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import cebra
from cebra import CEBRA

from hold_out import hold_out


PARAMETER_SETS = {
    "set0222": {"learning_rate": 0.0035, "min_temperature": 2.33, "max_iterations": 50000, "distance": "euclidean", "temp_mode": "constant"},
    "set0307": {"learning_rate": 0.007, "min_temperature": 1.75, "max_iterations": 7500, "distance": "cosine", "temp_mode": "constant"},
    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": "cosine", "temp_mode": "auto"},
    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": "euclidean", "temp_mode": "constant"},
    "set0816": {"learning_rate": 0.0095, "min_temperature": 2.66, "max_iterations": 25000, "distance": "cosine", "temp_mode": "constant"},
}

LABEL_COLORS = {
    1: "#d84a3a",
    2: "#f39c3d",
    3: "#f0c94a",
    4: "#c9d84a",
    5: "#4ca45c",
    6: "#36b39c",
    7: "#3c78d8",
    8: "#4d5bd6",
    9: "#7b4ab8",
    10: "#c04a9f",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single 10-bin conditional decoding iteration and save visualization artifacts.")
    parser.add_argument("traceA1An_An", type=str)
    parser.add_argument("traceAnB1_An", type=str)
    parser.add_argument("traceA1An_A1", type=str)
    parser.add_argument("traceAnB1_B1", type=str)
    parser.add_argument("CSUSAn", type=str)
    parser.add_argument("CSUSA1", type=str)
    parser.add_argument("CSUSB1", type=str)
    parser.add_argument("how_many_divisions", type=int, choices=[10], help="This script is intentionally limited to the 10-bin workflow.")
    parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1])
    parser.add_argument("--parameter_set_name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
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


def load_label(path: str) -> np.ndarray:
    return cebra.load_data(file=path)[0, :].flatten()


def keep_ten_bins(labels: np.ndarray) -> np.ndarray:
    binned = labels.copy()
    binned[binned == -1] = 0
    return binned.astype(int)


def filter_pretrial(trace: np.ndarray, labels: np.ndarray, include_pretrial: bool) -> Tuple[np.ndarray, np.ndarray]:
    keep_mask = labels != 0 if include_pretrial else labels > 0
    return trace[keep_mask], labels[keep_mask]


def trim_modulo_nine(trace_an_a1: np.ndarray, trace_an_b1: np.ndarray, labels_an: np.ndarray, trace_b1: np.ndarray, labels_b1: np.ndarray):
    if len(labels_an) % 10 == 9:
        labels_an = labels_an[9:]
        trace_an_a1 = trace_an_a1[9:]
        trace_an_b1 = trace_an_b1[9:]

    if len(labels_b1) % 10 == 9:
        labels_b1 = labels_b1[9:]
        trace_b1 = trace_b1[9:]

    return trace_an_a1, trace_an_b1, labels_an, trace_b1, labels_b1


def create_model(parameter_set: Dict[str, object]) -> CEBRA:
    common_kwargs = dict(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=parameter_set["learning_rate"],
        output_dimension=3,
        max_iterations=parameter_set["max_iterations"],
        distance=parameter_set["distance"],
        conditional="time_delta",
        device="cuda_if_available",
        num_hidden_units=32,
        time_offsets=1,
        verbose=True,
    )

    if parameter_set["temp_mode"] == "auto":
        return CEBRA(min_temperature=parameter_set["min_temperature"], **common_kwargs)

    return CEBRA(
        temperature_mode="constant",
        temperature=parameter_set["min_temperature"],
        **common_kwargs,
    )


def scatter_embedding(ax, embedding: np.ndarray, labels: np.ndarray, title: str):
    for label_value in sorted(np.unique(labels)):
        color = LABEL_COLORS.get(int(label_value), "#888888")
        mask = labels == label_value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=8,
            c=color,
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")


def build_figure(artifacts: Dict[str, Dict[str, np.ndarray]], figure_path: Path):
    fig = plt.figure(figsize=(18, 8))
    titles = [
        ("b_train", "Model from A(n) training data"),
        ("b_holdout", "Model applied to A(n) held-out data"),
        ("b_external", "Model applied to B1"),
    ]

    for idx, (key, title) in enumerate(titles, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        scatter_embedding(ax, artifacts[key]["embedding"], artifacts[key]["labels"], title)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"Bin {label}", markerfacecolor=color, markersize=7)
        for label, color in LABEL_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figure_path, format=figure_path.suffix.lstrip("."), dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.parameter_set_name not in PARAMETER_SETS:
        raise KeyError(f"Unknown parameter_set_name '{args.parameter_set_name}'. Available: {sorted(PARAMETER_SETS)}")

    parameter_set = PARAMETER_SETS[args.parameter_set_name]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{args.parameter_set_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_a1an_an = load_trace(args.traceA1An_An)
    trace_anb1_an = load_trace(args.traceAnB1_An)
    trace_a1an_a1 = load_trace(args.traceA1An_A1)
    trace_anb1_b1 = load_trace(args.traceAnB1_B1)

    labels_an = load_label(args.CSUSAn)
    labels_a1 = load_label(args.CSUSA1)
    labels_b1 = load_label(args.CSUSB1)

    include_pretrial = bool(args.pretrial_y_or_n)
    trace_a1an_an, labels_an = filter_pretrial(trace_a1an_an, labels_an, include_pretrial)
    trace_anb1_an, labels_an_for_b = filter_pretrial(trace_anb1_an, load_label(args.CSUSAn), include_pretrial)
    trace_a1an_a1, labels_a1 = filter_pretrial(trace_a1an_a1, labels_a1, include_pretrial)
    trace_anb1_b1, labels_b1 = filter_pretrial(trace_anb1_b1, labels_b1, include_pretrial)

    labels_an = keep_ten_bins(labels_an)
    labels_an_for_b = keep_ten_bins(labels_an_for_b)
    labels_a1 = keep_ten_bins(labels_a1)
    labels_b1 = keep_ten_bins(labels_b1)

    trace_a1an_an, trace_anb1_an, labels_an, trace_anb1_b1, labels_b1 = trim_modulo_nine(
        trace_a1an_an,
        trace_anb1_an,
        labels_an,
        trace_anb1_b1,
        labels_b1,
    )
    labels_an_for_b = labels_an.copy()

    trace_b_train, trace_b_holdout = hold_out(trace_anb1_an, 75)
    labels_b_train, labels_b_holdout = hold_out(labels_an_for_b, 75)

    print("Starting B1-matched model training...")
    model_b = create_model(parameter_set)
    model_b.fit(trace_b_train, labels_b_train)

    emb_b_train = model_b.transform(trace_b_train)
    emb_b_holdout = model_b.transform(trace_b_holdout)
    emb_b_external = model_b.transform(trace_anb1_b1)

    artifacts = {
        "b_train": {"embedding": emb_b_train, "labels": labels_b_train},
        "b_holdout": {"embedding": emb_b_holdout, "labels": labels_b_holdout},
        "b_external": {"embedding": emb_b_external, "labels": labels_b1},
    }

    figure_path = run_dir / "cond10_single_iteration_embeddings.svg"
    build_figure(artifacts, figure_path)

    np.savez_compressed(
        run_dir / "embeddings_and_labels.npz",
        emb_b_train=emb_b_train,
        labels_b_train=labels_b_train,
        emb_b_holdout=emb_b_holdout,
        labels_b_holdout=labels_b_holdout,
        emb_b_external=emb_b_external,
        labels_b1=labels_b1,
    )

    model_b_path = run_dir / "model_b1_matched.pt"
    model_b.save(str(model_b_path))

    metadata = {
        "timestamp": timestamp,
        "parameter_set_name": args.parameter_set_name,
        "how_many_divisions": args.how_many_divisions,
        "pretrial_y_or_n": args.pretrial_y_or_n,
        "seed": args.seed,
        "train_samples": int(trace_b_train.shape[0]),
        "holdout_samples": int(trace_b_holdout.shape[0]),
        "external_samples": int(trace_anb1_b1.shape[0]),
        "parameter_set": parameter_set,
        "traceAnB1_An": str(Path(args.traceAnB1_An).resolve()),
        "traceAnB1_B1": str(Path(args.traceAnB1_B1).resolve()),
        "eyeblinkAn": str(Path(args.CSUSAn).resolve()),
        "eyeblinkB1": str(Path(args.CSUSB1).resolve()),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
