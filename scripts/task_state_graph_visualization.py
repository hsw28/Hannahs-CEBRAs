#!/usr/bin/env python3
"""
Graph/topology-style visualization of CSUS task-state organization.

This graph visualization is a simplified abstraction of the task-state RDM.
It tests whether the neighborhood and sequential adjacency structure among
task states is preserved across animals. It should be interpreted together
with the full RDM/geometry-preservation analysis, which remains the primary
quantitative test of conserved relational organization.

Default use case:
    python scripts/task_state_graph_visualization.py

By default the script looks for the latest
geometry/procrustes_disparity_outputs/cross_rat_mean_ab/cross_rat_mean_ab_procrustes_*.csv
file, computes CSUS5 task-bin RDMs from the aligned coordinates, builds kNN
graphs, prints stats, and saves compact SVG/PNG panels.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_rel, wilcoxon


@dataclass
class TaskStateGraph:
    name: str
    rdm: np.ndarray
    adjacency: np.ndarray
    directed_knn_adjacency: np.ndarray
    nearest_neighbor: np.ndarray
    nearest_distance: np.ndarray
    sequential: Dict[str, Any]
    labels: List[str]
    coords: Optional[np.ndarray] = None
    shuffle_permutation: Optional[np.ndarray] = None


def sem(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def right_tail_p(real_value: float, shuffle_values: Sequence[float]) -> float:
    shuffle_values = np.asarray(shuffle_values, dtype=float)
    shuffle_values = shuffle_values[np.isfinite(shuffle_values)]
    if not np.isfinite(real_value) or len(shuffle_values) == 0:
        return np.nan
    return float((1 + np.sum(shuffle_values >= real_value)) / (len(shuffle_values) + 1))


def paired_tests(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return {"paired_t_stat": np.nan, "paired_t_p": np.nan, "wilcoxon_p": np.nan}

    t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
    try:
        _, w_p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_p = np.nan
    return {"paired_t_stat": float(t_stat), "paired_t_p": float(t_p), "wilcoxon_p": float(w_p)}


def classical_mds(rdm: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS from a square distance matrix, with no sklearn dependency."""
    distance = np.asarray(rdm, dtype=float)
    n = distance.shape[0]
    distance = np.nan_to_num(distance, nan=np.nanmean(distance[np.isfinite(distance)]))
    np.fill_diagonal(distance, 0.0)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    gram = (gram + gram.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order[:n_components]], 0.0)
    eigvecs = eigvecs[:, order[:n_components]]
    coords = eigvecs * np.sqrt(eigvals)
    if coords.shape[1] < n_components:
        coords = np.column_stack([coords, np.zeros((n, n_components - coords.shape[1]))])
    return normalize_coords(coords)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    coords = coords[:, :2] if coords.shape[1] >= 2 else np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    coords = coords - np.nanmean(coords, axis=0, keepdims=True)
    scale = np.nanmax(np.abs(coords))
    if np.isfinite(scale) and scale > 0:
        coords = coords / scale
    return coords


def build_task_state_graph_from_rdm(rdm: np.ndarray, k_nearest: int = 1) -> Dict[str, Any]:
    rdm = np.asarray(rdm, dtype=float)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(f"rdm must be square, got {rdm.shape}.")

    n_bins = rdm.shape[0]
    k = int(np.clip(k_nearest, 1, max(1, n_bins - 1)))
    rdm_no_self = rdm.copy()
    np.fill_diagonal(rdm_no_self, np.inf)

    directed = np.zeros((n_bins, n_bins), dtype=bool)
    nearest_neighbor = np.full(n_bins, -1, dtype=int)
    nearest_distance = np.full(n_bins, np.nan, dtype=float)
    for row in range(n_bins):
        order = np.argsort(rdm_no_self[row])
        order = order[np.isfinite(rdm_no_self[row, order])]
        if len(order) == 0:
            continue
        nearest_neighbor[row] = int(order[0])
        nearest_distance[row] = float(rdm_no_self[row, order[0]])
        directed[row, order[:k]] = True

    adjacency = directed | directed.T
    np.fill_diagonal(adjacency, False)
    sequential = sequential_adjacency_metrics(rdm)
    return {
        "rdm": rdm,
        "adjacency": adjacency,
        "directed_knn_adjacency": directed,
        "nearest_neighbor": nearest_neighbor,
        "nearest_distance": nearest_distance,
        "sequential": sequential,
        "k_nearest": k,
    }


def sequential_adjacency_metrics(rdm: np.ndarray) -> Dict[str, Any]:
    n_bins = rdm.shape[0]
    adjacent_pairs = np.array([(idx, idx + 1) for idx in range(n_bins - 1)], dtype=int)
    upper = np.triu(np.ones((n_bins, n_bins), dtype=bool), k=1)
    adjacent_mask = np.zeros((n_bins, n_bins), dtype=bool)
    for left, right in adjacent_pairs:
        adjacent_mask[left, right] = True
        adjacent_mask[right, left] = True

    adjacent_distances = rdm[upper & adjacent_mask]
    nonadjacent_distances = rdm[upper & ~adjacent_mask]
    if len(adjacent_distances) and len(nonadjacent_distances):
        comparisons = adjacent_distances[:, None] < nonadjacent_distances[None, :]
        fraction_adjacent_closer = float(np.mean(comparisons))
    else:
        fraction_adjacent_closer = np.nan

    mean_adjacent = float(np.nanmean(adjacent_distances)) if len(adjacent_distances) else np.nan
    mean_nonadjacent = float(np.nanmean(nonadjacent_distances)) if len(nonadjacent_distances) else np.nan
    return {
        "adjacent_pairs": adjacent_pairs,
        "adjacent_distances": adjacent_distances,
        "nonadjacent_distances": nonadjacent_distances,
        "mean_adjacent_distance": mean_adjacent,
        "mean_nonadjacent_distance": mean_nonadjacent,
        "nonadjacent_minus_adjacent": mean_nonadjacent - mean_adjacent,
        "fraction_adjacent_closer_than_nonadjacent": fraction_adjacent_closer,
    }


def make_task_state_graph(
    name: str,
    rdm: np.ndarray,
    labels: Sequence[str],
    coords: Optional[np.ndarray] = None,
    k_nearest: int = 1,
) -> TaskStateGraph:
    graph = build_task_state_graph_from_rdm(rdm, k_nearest=k_nearest)
    return TaskStateGraph(
        name=name,
        rdm=graph["rdm"],
        adjacency=graph["adjacency"],
        directed_knn_adjacency=graph["directed_knn_adjacency"],
        nearest_neighbor=graph["nearest_neighbor"],
        nearest_distance=graph["nearest_distance"],
        sequential=graph["sequential"],
        labels=list(labels),
        coords=None if coords is None else np.asarray(coords, dtype=float),
    )


def adjacency_similarity(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    mask = np.triu(np.ones_like(left, dtype=bool), k=1)
    a = left[mask].astype(bool)
    b = right[mask].astype(bool)
    agreement = float(np.mean(a == b)) if len(a) else np.nan
    union = np.sum(a | b)
    jaccard = float(np.sum(a & b) / union) if union > 0 else np.nan
    return agreement, jaccard


def mean_rdm(graphs: Sequence[TaskStateGraph]) -> np.ndarray:
    stack = np.stack([graph.rdm for graph in graphs], axis=0)
    out = np.nanmean(stack, axis=0)
    np.fill_diagonal(out, 0.0)
    return out


def compute_graph_preservation_across_rats(
    graphs: Sequence[TaskStateGraph],
    k_nearest: int = 1,
    n_shuffles: int = 500,
    random_seed: int = 1,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    if not graphs:
        raise ValueError("At least one graph is required.")

    group_graph = make_task_state_graph(
        "group_average",
        mean_rdm(graphs),
        graphs[0].labels,
        coords=None,
        k_nearest=k_nearest,
    )
    n_bins = group_graph.rdm.shape[0]

    nn_match = np.zeros((len(graphs), n_bins), dtype=bool)
    edge_agreement = np.full(len(graphs), np.nan)
    edge_jaccard = np.full(len(graphs), np.nan)
    mean_adjacent = np.full(len(graphs), np.nan)
    mean_nonadjacent = np.full(len(graphs), np.nan)
    nonadjacent_minus_adjacent = np.full(len(graphs), np.nan)
    fraction_adjacent_closer = np.full(len(graphs), np.nan)

    for idx, graph in enumerate(graphs):
        nn_match[idx] = graph.nearest_neighbor == group_graph.nearest_neighbor
        edge_agreement[idx], edge_jaccard[idx] = adjacency_similarity(graph.adjacency, group_graph.adjacency)
        mean_adjacent[idx] = graph.sequential["mean_adjacent_distance"]
        mean_nonadjacent[idx] = graph.sequential["mean_nonadjacent_distance"]
        nonadjacent_minus_adjacent[idx] = graph.sequential["nonadjacent_minus_adjacent"]
        fraction_adjacent_closer[idx] = graph.sequential["fraction_adjacent_closer_than_nonadjacent"]

    shuffle_nn = np.full(n_shuffles, np.nan)
    shuffle_edge = np.full(n_shuffles, np.nan)
    shuffle_jaccard = np.full(n_shuffles, np.nan)
    shuffle_seq = np.full(n_shuffles, np.nan)
    shuffle_example_candidates = []

    for shuffle_idx in range(n_shuffles):
        nn_scores = []
        edge_scores = []
        jaccard_scores = []
        seq_scores = []
        for graph_idx, graph in enumerate(graphs):
            perm = rng.permutation(n_bins)
            shuffled_rdm = graph.rdm[perm][:, perm]
            shuffled_graph = make_task_state_graph(
                f"{graph.name}_shuffle",
                shuffled_rdm,
                graph.labels,
                coords=None,
                k_nearest=k_nearest,
            )
            shuffled_graph.shuffle_permutation = perm
            nn_scores.append(np.mean(shuffled_graph.nearest_neighbor == group_graph.nearest_neighbor))
            agreement, jaccard = adjacency_similarity(shuffled_graph.adjacency, group_graph.adjacency)
            edge_scores.append(agreement)
            jaccard_scores.append(jaccard)
            seq_scores.append(shuffled_graph.sequential["nonadjacent_minus_adjacent"])
            if graph_idx == 0:
                shuffle_example_candidates.append(
                    {
                        "shuffle_idx": shuffle_idx,
                        "graph": shuffled_graph,
                        "edge_agreement": agreement,
                        "nn_preservation": float(nn_scores[-1]),
                        "nonadjacent_minus_adjacent": shuffled_graph.sequential["nonadjacent_minus_adjacent"],
                    }
                )
        shuffle_nn[shuffle_idx] = np.nanmean(nn_scores)
        shuffle_edge[shuffle_idx] = np.nanmean(edge_scores)
        shuffle_jaccard[shuffle_idx] = np.nanmean(jaccard_scores)
        shuffle_seq[shuffle_idx] = np.nanmean(seq_scores)

    shuffle_example = choose_median_shuffle_example(shuffle_example_candidates, shuffle_edge)

    seq_tests = paired_tests(mean_nonadjacent, mean_adjacent)
    results = {
        "group_graph": group_graph,
        "nearest_neighbor": {
            "match_matrix": nn_match,
            "rat_preservation": np.mean(nn_match, axis=1),
            "bin_preservation": np.mean(nn_match, axis=0),
            "mean_preservation": float(np.mean(nn_match)),
            "shuffle_p": right_tail_p(float(np.mean(nn_match)), shuffle_nn),
        },
        "graph_similarity": {
            "edge_agreement": edge_agreement,
            "edge_jaccard": edge_jaccard,
            "mean_edge_agreement": float(np.nanmean(edge_agreement)),
            "mean_edge_jaccard": float(np.nanmean(edge_jaccard)),
            "shuffle_p": right_tail_p(float(np.nanmean(edge_agreement)), shuffle_edge),
        },
        "sequential": {
            "mean_adjacent_distance": mean_adjacent,
            "mean_nonadjacent_distance": mean_nonadjacent,
            "nonadjacent_minus_adjacent": nonadjacent_minus_adjacent,
            "fraction_adjacent_closer_than_nonadjacent": fraction_adjacent_closer,
            "mean_nonadjacent_minus_adjacent": float(np.nanmean(nonadjacent_minus_adjacent)),
            "sem_nonadjacent_minus_adjacent": sem(nonadjacent_minus_adjacent),
            "shuffle_p": right_tail_p(float(np.nanmean(nonadjacent_minus_adjacent)), shuffle_seq),
            **seq_tests,
        },
        "shuffle": {
            "mean_nearest_neighbor_preservation": shuffle_nn,
            "mean_graph_similarity": shuffle_edge,
            "mean_graph_jaccard": shuffle_jaccard,
            "mean_nonadjacent_minus_adjacent": shuffle_seq,
            "example_graph": shuffle_example,
        },
    }
    return results


def choose_median_shuffle_example(
    candidates: Sequence[Mapping[str, Any]],
    shuffle_edge: Sequence[float],
) -> Optional[TaskStateGraph]:
    """Pick a reproducible representative shuffle near the null median.

    The summary plot already shows the full null distribution. The node-link
    control panel is only an example, so choose the first-rat shuffled graph
    from the shuffle iteration whose mean edge agreement is closest to the
    median null edge agreement, rather than a random or extreme example.
    """
    if not candidates:
        return None
    shuffle_edge = np.asarray(shuffle_edge, dtype=float)
    median_edge = float(np.nanmedian(shuffle_edge))
    best_candidate = None
    best_key = None
    for candidate in candidates:
        shuffle_idx = int(candidate["shuffle_idx"])
        mean_edge = shuffle_edge[shuffle_idx]
        if not np.isfinite(mean_edge):
            continue
        key = (
            abs(mean_edge - median_edge),
            abs(float(candidate["edge_agreement"]) - median_edge),
            abs(float(candidate["nn_preservation"]) - np.nanmedian([c["nn_preservation"] for c in candidates])),
            shuffle_idx,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_candidate = candidate
    if best_candidate is None:
        return None
    graph = best_candidate["graph"]
    graph.name = "median_shuffle_control"
    return graph


def load_graphs_from_aligned_csv(
    csv_path: Path,
    *,
    k_nearest: int = 1,
    task_bins: Sequence[int] = (1, 2, 3, 4, 5),
    environment: Optional[str] = None,
    coord_columns: Optional[Sequence[str]] = None,
) -> List[TaskStateGraph]:
    df = pd.read_csv(csv_path)
    required = {"rat_id", "task_bin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if environment is not None and "environment" in df.columns:
        df = df[df["environment"].astype(str) == str(environment)].copy()

    task_bins = list(task_bins)
    df = df[df["task_bin"].isin(task_bins)].copy()
    if df.empty:
        raise ValueError(f"No requested task bins {task_bins} found in {csv_path}.")

    if coord_columns is None:
        coord_columns = [col for col in ("x", "y", "z") if col in df.columns]
        if len(coord_columns) < 2:
            dim_cols = sorted([col for col in df.columns if col.startswith("dim")], key=lambda c: int(c[3:]))
            coord_columns = dim_cols
    if len(coord_columns) < 2:
        raise ValueError("Need at least two coordinate columns, e.g. x/y or dim1/dim2.")

    labels = [f"CSUS5 bin {int(bin_value)}" for bin_value in task_bins]
    graphs = []
    for rat_id, subset in df.groupby("rat_id", sort=True):
        subset = subset.sort_values("task_bin")
        available_bins = subset["task_bin"].astype(int).tolist()
        if available_bins != task_bins:
            raise ValueError(f"{rat_id} has bins {available_bins}; expected {task_bins}.")
        coords = subset[list(coord_columns)].to_numpy(dtype=float)
        rdm = squareform(pdist(coords, metric="euclidean"))
        graphs.append(make_task_state_graph(str(rat_id), rdm, labels, coords=coords, k_nearest=k_nearest))
    return graphs


def latest_aligned_csv(root: Path) -> Path:
    search_dir = root / "geometry" / "procrustes_disparity_outputs" / "cross_rat_mean_ab"
    candidates = sorted(search_dir.glob("cross_rat_mean_ab_procrustes_*.csv"))
    candidates = [path for path in candidates if "pca2d" not in path.name and "disparity" not in path.name]
    if not candidates:
        raise FileNotFoundError(
            "No cross_rat_mean_ab_procrustes_*.csv file found under "
            f"{search_dir}. Pass --aligned_csv to use a file elsewhere."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def panel_positions(graphs: Sequence[TaskStateGraph], results: Mapping[str, Any], use_aligned_coords: bool) -> Dict[str, np.ndarray]:
    group_coords = classical_mds(results["group_graph"].rdm, n_components=2)
    shuffle_graph = results["shuffle"]["example_graph"]
    if shuffle_graph is not None and shuffle_graph.shuffle_permutation is not None:
        shuffled_coords = group_coords[np.asarray(shuffle_graph.shuffle_permutation, dtype=int)]
    else:
        shuffled_coords = group_coords
    positions = {"group_average": group_coords, "shuffled_control": shuffled_coords}
    for graph in graphs:
        if use_aligned_coords and graph.coords is not None:
            positions[graph.name] = normalize_coords(graph.coords)
        else:
            positions[graph.name] = group_coords
    return positions


def draw_one_graph(
    ax: plt.Axes,
    graph: TaskStateGraph,
    coords: np.ndarray,
    *,
    title: str,
    show_nearest_edges: bool = True,
    show_sequence_arrows: bool = True,
    node_colors: Optional[np.ndarray] = None,
) -> None:
    n_bins = graph.rdm.shape[0]
    if node_colors is None:
        node_colors = plt.cm.viridis(np.linspace(0.12, 0.9, n_bins))

    ax.set_aspect("equal")
    ax.axis("off")

    if show_nearest_edges:
        rows, cols = np.where(np.triu(graph.adjacency, k=1))
        for left, right in zip(rows, cols):
            ax.plot(
                [coords[left, 0], coords[right, 0]],
                [coords[left, 1], coords[right, 1]],
                color="#b7b7b7",
                linewidth=1.2,
                zorder=1,
            )

    if show_sequence_arrows:
        for idx in range(n_bins - 1):
            start = coords[idx]
            end = coords[idx + 1]
            delta = end - start
            arrow = FancyArrowPatch(
                start + 0.18 * delta,
                start + 0.82 * delta,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#1a1a1a",
                zorder=2,
            )
            ax.add_patch(arrow)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=230,
        c=node_colors,
        edgecolors="#1f1f1f",
        linewidths=0.8,
        zorder=3,
    )
    for idx, label in enumerate(graph.labels):
        ax.text(
            coords[idx, 0],
            coords[idx, 1],
            str(idx + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            coords[idx, 0],
            coords[idx, 1] - 0.15,
            label.replace("CSUS5 ", ""),
            ha="center",
            va="top",
            color="#242424",
            fontsize=7,
            zorder=4,
        )
    pad = 0.35
    ax.set_xlim(np.nanmin(coords[:, 0]) - pad, np.nanmax(coords[:, 0]) + pad)
    ax.set_ylim(np.nanmin(coords[:, 1]) - pad, np.nanmax(coords[:, 1]) + pad)
    ax.set_title(title, fontsize=10, fontweight="normal")


def plot_task_state_graphs(
    graphs: Sequence[TaskStateGraph],
    results: Mapping[str, Any],
    *,
    output_path: Optional[Path] = None,
    use_aligned_coords: bool = True,
    max_rat_graphs: Optional[int] = None,
) -> plt.Figure:
    max_count = len(graphs) if max_rat_graphs is None else min(len(graphs), max_rat_graphs)
    shown_graphs = list(graphs[:max_count])
    group_graph = results["group_graph"]
    shuffle_graph = results["shuffle"]["example_graph"]
    positions = panel_positions(graphs, results, use_aligned_coords=use_aligned_coords)

    n_panels = len(shown_graphs) + 2
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.15 * n_cols, 2.95 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, graph in enumerate(shown_graphs):
        draw_one_graph(axes_flat[idx], graph, positions[graph.name], title=graph.name)
    draw_one_graph(
        axes_flat[len(shown_graphs)],
        group_graph,
        positions["group_average"],
        title="group average",
    )
    if shuffle_graph is not None:
        shuffle_title = "shuffled control"
        if shuffle_graph.shuffle_permutation is not None:
            one_based_perm = np.asarray(shuffle_graph.shuffle_permutation, dtype=int) + 1
            shuffle_title = "shuffled labels (" + ",".join(str(x) for x in one_based_perm) + ")"
        draw_one_graph(
            axes_flat[len(shown_graphs) + 1],
            shuffle_graph,
            positions["shuffled_control"],
            title=shuffle_title,
        )
    else:
        axes_flat[len(shown_graphs) + 1].axis("off")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.suptitle("Shared CSUS5 task-state organization", fontsize=12, fontweight="normal")
    fig.tight_layout()
    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def scatter_with_mean(ax: plt.Axes, x: float, values: Sequence[float], color: str, marker: str = "o") -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(np.full(len(values), x) + jitter, values, s=28, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    mean = np.nanmean(values)
    err = sem(values)
    ax.scatter([x], [mean], s=70, color=color, marker=marker, edgecolor="black", linewidth=0.8, zorder=4)
    if np.isfinite(err):
        ax.plot([x, x], [mean - err, mean + err], color="black", linewidth=1.2, zorder=4)


def plot_graph_summary_metrics(results: Mapping[str, Any], output_path: Optional[Path] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35))

    adjacent = results["sequential"]["mean_adjacent_distance"]
    nonadjacent = results["sequential"]["mean_nonadjacent_distance"]
    for idx in range(len(adjacent)):
        axes[0].plot([1, 2], [adjacent[idx], nonadjacent[idx]], color="#b8b8b8", linewidth=0.9, zorder=1)
    scatter_with_mean(axes[0], 1, adjacent, "#2d7fb8", marker="s")
    scatter_with_mean(axes[0], 2, nonadjacent, "#d65f2e", marker="s")
    axes[0].set_xticks([1, 2], ["adjacent", "non-adjacent"])
    axes[0].set_xlim(0.55, 2.45)
    axes[0].set_ylabel("Mean RDM distance")
    axes[0].set_title(f"Adjacency p={results['sequential']['paired_t_p']:.3g}", fontweight="normal")

    real_nn = results["nearest_neighbor"]["rat_preservation"]
    shuffle_nn = results["shuffle"]["mean_nearest_neighbor_preservation"]
    scatter_with_mean(axes[1], 1, shuffle_nn[: min(500, len(shuffle_nn))], "#b8b8b8", marker="s")
    scatter_with_mean(axes[1], 2, real_nn, "#2d7fb8", marker="s")
    axes[1].set_xticks([1, 2], ["shuffle", "real"])
    axes[1].set_xlim(0.55, 2.45)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("NN match fraction")
    axes[1].set_title(f"NN preservation p={results['nearest_neighbor']['shuffle_p']:.3g}", fontweight="normal")

    real_graph = results["graph_similarity"]["edge_agreement"]
    shuffle_graph = results["shuffle"]["mean_graph_similarity"]
    scatter_with_mean(axes[2], 1, shuffle_graph[: min(500, len(shuffle_graph))], "#b8b8b8", marker="s")
    scatter_with_mean(axes[2], 2, real_graph, "#388a45", marker="s")
    axes[2].set_xticks([1, 2], ["shuffle", "real"])
    axes[2].set_xlim(0.55, 2.45)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Graph edge agreement")
    axes[2].set_title(f"Graph similarity p={results['graph_similarity']['shuffle_p']:.3g}", fontweight="normal")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
        ax.grid(axis="y", color="#e7e7e7", linewidth=0.7)
    fig.suptitle("CSUS5 graph preservation summary", fontsize=12, fontweight="normal")
    fig.tight_layout()
    if output_path is not None:
        save_figure(fig, output_path)
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() == ".png":
        kwargs["dpi"] = 300
    fig.savefig(output_path, **kwargs)


def metric_rows(graphs: Sequence[TaskStateGraph], results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for idx, graph in enumerate(graphs):
        rows.append(
            {
                "rat_id": graph.name,
                "nearest_neighbor_preservation": results["nearest_neighbor"]["rat_preservation"][idx],
                "graph_edge_agreement": results["graph_similarity"]["edge_agreement"][idx],
                "graph_edge_jaccard": results["graph_similarity"]["edge_jaccard"][idx],
                "mean_adjacent_distance": results["sequential"]["mean_adjacent_distance"][idx],
                "mean_nonadjacent_distance": results["sequential"]["mean_nonadjacent_distance"][idx],
                "nonadjacent_minus_adjacent": results["sequential"]["nonadjacent_minus_adjacent"][idx],
                "fraction_adjacent_closer_than_nonadjacent": results["sequential"][
                    "fraction_adjacent_closer_than_nonadjacent"
                ][idx],
            }
        )
    return pd.DataFrame(rows)


def summary_row(results: Mapping[str, Any], source_csv: Optional[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_csv": "" if source_csv is None else str(source_csv),
                "mean_nearest_neighbor_preservation": results["nearest_neighbor"]["mean_preservation"],
                "nearest_neighbor_shuffle_p": results["nearest_neighbor"]["shuffle_p"],
                "mean_graph_edge_agreement": results["graph_similarity"]["mean_edge_agreement"],
                "mean_graph_edge_jaccard": results["graph_similarity"]["mean_edge_jaccard"],
                "graph_similarity_shuffle_p": results["graph_similarity"]["shuffle_p"],
                "mean_nonadjacent_minus_adjacent": results["sequential"]["mean_nonadjacent_minus_adjacent"],
                "sem_nonadjacent_minus_adjacent": results["sequential"]["sem_nonadjacent_minus_adjacent"],
                "adjacent_vs_nonadjacent_paired_t_p": results["sequential"]["paired_t_p"],
                "adjacent_vs_nonadjacent_wilcoxon_p": results["sequential"]["wilcoxon_p"],
                "sequential_shuffle_p": results["sequential"]["shuffle_p"],
            }
        ]
    )


def print_results(results: Mapping[str, Any]) -> None:
    print("\nTask-state graph preservation summary")
    print(f"  Mean NN preservation vs group: {results['nearest_neighbor']['mean_preservation']:.3f}")
    print(
        "  Shuffle NN preservation: "
        f"{np.nanmean(results['shuffle']['mean_nearest_neighbor_preservation']):.3f} +/- "
        f"{np.nanstd(results['shuffle']['mean_nearest_neighbor_preservation']):.3f}, "
        f"p={results['nearest_neighbor']['shuffle_p']:.4g}"
    )
    print(f"  Mean graph edge agreement vs group: {results['graph_similarity']['mean_edge_agreement']:.3f}")
    print(
        "  Shuffle graph edge agreement: "
        f"{np.nanmean(results['shuffle']['mean_graph_similarity']):.3f} +/- "
        f"{np.nanstd(results['shuffle']['mean_graph_similarity']):.3f}, "
        f"p={results['graph_similarity']['shuffle_p']:.4g}"
    )
    print(
        "  Adjacent distance: "
        f"{np.nanmean(results['sequential']['mean_adjacent_distance']):.3f} +/- "
        f"{sem(results['sequential']['mean_adjacent_distance']):.3f}"
    )
    print(
        "  Non-adjacent distance: "
        f"{np.nanmean(results['sequential']['mean_nonadjacent_distance']):.3f} +/- "
        f"{sem(results['sequential']['mean_nonadjacent_distance']):.3f}"
    )
    print(
        "  Adjacent-vs-nonadjacent paired t-test "
        f"p={results['sequential']['paired_t_p']:.4g}, "
        f"Wilcoxon p={results['sequential']['wilcoxon_p']:.4g}, "
        f"shuffle p={results['sequential']['shuffle_p']:.4g}"
    )
    print(
        "  Interpretation: positive nonadjacent-minus-adjacent values mean "
        "sequentially adjacent CSUS bins are closer in the task-state RDM."
    )


def run_task_state_graph_visualization(
    graphs: Sequence[TaskStateGraph],
    *,
    k_nearest: int = 1,
    n_shuffles: int = 500,
    random_seed: int = 1,
    output_dir: Optional[Path] = None,
    save_figs: bool = True,
    use_aligned_coords: bool = True,
    source_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    results = compute_graph_preservation_across_rats(
        graphs,
        k_nearest=k_nearest,
        n_shuffles=n_shuffles,
        random_seed=random_seed,
    )
    print_results(results)

    figures = {}
    if save_figs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_svg = output_dir / "task_state_graphs.svg"
        graph_png = output_dir / "task_state_graphs.png"
        summary_svg = output_dir / "task_state_graph_summary.svg"
        summary_png = output_dir / "task_state_graph_summary.png"

        fig = plot_task_state_graphs(graphs, results, output_path=graph_svg, use_aligned_coords=use_aligned_coords)
        save_figure(fig, graph_png)
        figures["graphs"] = fig

        fig = plot_graph_summary_metrics(results, output_path=summary_svg)
        save_figure(fig, summary_png)
        figures["summary"] = fig

        metric_rows(graphs, results).to_csv(output_dir / "task_state_graph_metrics_by_rat.csv", index=False)
        summary_row(results, source_csv).to_csv(output_dir / "task_state_graph_summary_metrics.csv", index=False)
        np.savez(
            output_dir / "task_state_graph_results.npz",
            group_rdm=results["group_graph"].rdm,
            nn_match_matrix=results["nearest_neighbor"]["match_matrix"],
            rat_nn_preservation=results["nearest_neighbor"]["rat_preservation"],
            edge_agreement=results["graph_similarity"]["edge_agreement"],
            edge_jaccard=results["graph_similarity"]["edge_jaccard"],
            mean_adjacent_distance=results["sequential"]["mean_adjacent_distance"],
            mean_nonadjacent_distance=results["sequential"]["mean_nonadjacent_distance"],
            shuffle_nn=results["shuffle"]["mean_nearest_neighbor_preservation"],
            shuffle_edge=results["shuffle"]["mean_graph_similarity"],
            shuffle_nonadjacent_minus_adjacent=results["shuffle"]["mean_nonadjacent_minus_adjacent"],
            shuffle_example_permutation=np.array([])
            if results["shuffle"]["example_graph"] is None
            else np.asarray(results["shuffle"]["example_graph"].shuffle_permutation),
        )
        print(f"\nSaved graph outputs to {output_dir}")
    else:
        figures["graphs"] = plot_task_state_graphs(graphs, results, use_aligned_coords=use_aligned_coords)
        figures["summary"] = plot_graph_summary_metrics(results)

    return {"graphs": list(graphs), "results": results, "figures": figures}


def parse_bins(text: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CSUS task-state graph/topology preservation from aligned CEBRA trajectories.")
    parser.add_argument("--aligned_csv", type=Path, default=None, help="Aligned per-rat task-bin CSV. Defaults to latest cross-rat mean-AB CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("geometry/task_state_graph_output"), help="Directory for figures and metric tables.")
    parser.add_argument("--task_bins", type=parse_bins, default=(1, 2, 3, 4, 5), help="Comma-separated task bins to include. Default: 1,2,3,4,5.")
    parser.add_argument("--environment", default=None, help="Optional environment column value to filter, e.g. A or B.")
    parser.add_argument("--coord_columns", default=None, help="Comma-separated coordinate columns. Default: x,y,z when present.")
    parser.add_argument("--k_nearest", type=int, default=1, help="k for k-nearest-neighbor graph edges.")
    parser.add_argument("--n_shuffles", type=int, default=500, help="Number of shuffled bin-label controls.")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for shuffle controls.")
    parser.add_argument("--use_mds_positions", action="store_true", help="Use common group-RDM MDS coordinates for all rat graph panels.")
    parser.add_argument("--no_save_figs", action="store_true", help="Create figures but do not save outputs.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aligned_csv = args.aligned_csv if args.aligned_csv is not None else latest_aligned_csv(repo_root)
    coord_columns = None
    if args.coord_columns:
        coord_columns = [part.strip() for part in args.coord_columns.split(",") if part.strip()]

    graphs = load_graphs_from_aligned_csv(
        aligned_csv,
        k_nearest=args.k_nearest,
        task_bins=args.task_bins,
        environment=args.environment,
        coord_columns=coord_columns,
    )
    print(f"Loaded {len(graphs)} rat graphs from {aligned_csv}")
    run_task_state_graph_visualization(
        graphs,
        k_nearest=args.k_nearest,
        n_shuffles=args.n_shuffles,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        save_figs=not args.no_save_figs,
        use_aligned_coords=not args.use_mds_positions,
        source_csv=aligned_csv,
    )


if __name__ == "__main__":
    main()
