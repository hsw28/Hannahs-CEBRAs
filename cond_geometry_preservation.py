import os
import sys
from datetime import datetime

sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, ttest_rel


def make_cebra_model(parameter_set, output_dimension):
    from cebra import CEBRA

    kwargs = {
        "model_architecture": "offset10-model",
        "batch_size": 512,
        "learning_rate": parameter_set["learning_rate"],
        "output_dimension": output_dimension,
        "max_iterations": parameter_set["max_iterations"],
        "distance": parameter_set["distance"],
        "conditional": "time_delta",
        "device": "cuda_if_available",
        "num_hidden_units": 32,
        "time_offsets": 1,
        "verbose": False,
    }

    if parameter_set["temp_mode"] == "auto":
        kwargs["temperature_mode"] = "auto"
        kwargs["min_temperature"] = parameter_set["min_temperature"]
    elif parameter_set["temp_mode"] == "constant":
        kwargs["temperature_mode"] = "constant"
        kwargs["temperature"] = parameter_set["min_temperature"]
    else:
        raise ValueError(f"Unknown temperature mode: {parameter_set['temp_mode']}")

    return CEBRA(**kwargs)


def trim_mod10_if_needed(traces, labels):
    min_length = len(labels)
    if min_length % 10 == 9:
        return traces[9:], labels[9:]
    return traces, labels


def common_bins(labels_a, labels_b):
    bins = np.intersect1d(np.unique(labels_a), np.unique(labels_b))
    bins = bins[bins != 0]
    if len(bins) < 2:
        raise ValueError("At least two shared task bins are required for a distance comparison.")
    if len(bins) < 3:
        print(
            "Warning: fewer than 3 shared task bins. Spearman geometry scores "
            "are undefined when there are fewer than 3 pairwise distances."
        )
    return bins


def bin_mean_embedding(embedding, labels, bins):
    rows = []
    for task_bin in bins:
        mask = labels == task_bin
        if not np.any(mask):
            raise ValueError(f"Task bin {task_bin} is missing from one environment.")
        rows.append(np.mean(embedding[mask], axis=0))
    return np.vstack(rows)


def zscore_columns(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def geometry_preservation_score(z_a, z_b):
    if z_a.shape != z_b.shape:
        raise ValueError(f"Embedding matrices must match, got {z_a.shape} and {z_b.shape}.")

    z_a = zscore_columns(z_a)
    z_b = zscore_columns(z_b)
    distances_a = pdist(z_a, metric="euclidean")
    distances_b = pdist(z_b, metric="euclidean")

    if len(distances_a) < 2:
        return np.nan

    score = spearmanr(distances_a, distances_b).correlation
    return score


def compute_geometry_preservation_run(z_a, z_b, n_shuff=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if n_shuff < 1:
        raise ValueError("n_shuff must be at least 1.")

    r_real = geometry_preservation_score(z_a, z_b)
    r_shuff_all = np.zeros(n_shuff)
    for shuffle_idx in range(n_shuff):
        permutation = rng.permutation(z_b.shape[0])
        r_shuff_all[shuffle_idx] = geometry_preservation_score(z_a, z_b[permutation])

    return {
        "rReal": r_real,
        "rShuff": np.nanmean(r_shuff_all),
        "rShuffAll": r_shuff_all,
    }


def paired_sign_flip_test(differences, n_permutations=None, rng=None):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if rng is None:
        rng = np.random.default_rng()
    if len(differences) == 0:
        return {"p_two_sided": np.nan, "observed_mean": np.nan, "n": 0, "n_permutations": 0, "exact": False}

    observed = np.mean(differences)
    n_runs = len(differences)
    if n_permutations is None and n_runs <= 20:
        signs = np.array(np.meshgrid(*([[-1, 1]] * n_runs))).T.reshape(-1, n_runs)
        null_means = np.mean(signs * differences, axis=1)
        exact = True
    else:
        if n_permutations is None:
            n_permutations = 10000
        signs = rng.choice([-1, 1], size=(n_permutations, n_runs))
        null_means = np.mean(signs * differences, axis=1)
        exact = False

    p_two_sided = (np.sum(np.abs(null_means) >= np.abs(observed)) + 1) / (len(null_means) + 1)
    return {
        "p_two_sided": p_two_sided,
        "observed_mean": observed,
        "n": n_runs,
        "n_permutations": len(null_means),
        "exact": exact,
    }


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(len(values))


def paired_geometry_stats(r_real, r_shuff, n_permutations=None, rng=None):
    r_real = np.asarray(r_real, dtype=float).reshape(-1)
    r_shuff = np.asarray(r_shuff, dtype=float).reshape(-1)
    if r_real.shape != r_shuff.shape:
        raise ValueError(f"rReal and rShuff must have the same shape, got {r_real.shape} and {r_shuff.shape}.")

    finite = np.isfinite(r_real) & np.isfinite(r_shuff)
    r_real = r_real[finite]
    r_shuff = r_shuff[finite]
    differences = r_real - r_shuff
    sign_flip = paired_sign_flip_test(differences, n_permutations=n_permutations, rng=rng)

    if len(differences) > 1:
        t_stat, t_p = ttest_rel(r_real, r_shuff, nan_policy="omit")
    else:
        t_stat, t_p = np.nan, np.nan

    return {
        "n_runs": len(differences),
        "real_mean": np.nanmean(r_real) if len(r_real) else np.nan,
        "real_sem": sem(r_real),
        "shuff_mean": np.nanmean(r_shuff) if len(r_shuff) else np.nan,
        "shuff_sem": sem(r_shuff),
        "diff_mean": np.nanmean(differences) if len(differences) else np.nan,
        "diff_sem": sem(differences),
        "sign_flip_p_two_sided": sign_flip["p_two_sided"],
        "sign_flip_n_permutations": sign_flip["n_permutations"],
        "sign_flip_exact": sign_flip["exact"],
        "paired_t_stat": t_stat,
        "paired_t_p_two_sided": t_p,
    }


def normalize_run_embeddings(embeddings):
    if isinstance(embeddings, np.ndarray):
        if embeddings.ndim == 2:
            return [embeddings]
        if embeddings.ndim == 3:
            return [embeddings[i] for i in range(embeddings.shape[0])]
    return list(embeddings)


def compute_geometry_preservation_group(
    z_a_runs,
    z_b_runs,
    n_shuff=1,
    n_permutations=None,
    random_seed=None,
    plot_path=None,
    title_suffix=None,
):
    rng = np.random.default_rng(random_seed)
    z_a_runs = normalize_run_embeddings(z_a_runs)
    z_b_runs = normalize_run_embeddings(z_b_runs)
    if len(z_a_runs) != len(z_b_runs):
        raise ValueError(f"Expected the same number of A and B runs, got {len(z_a_runs)} and {len(z_b_runs)}.")

    r_real = np.zeros(len(z_a_runs))
    r_shuff = np.zeros(len(z_a_runs))
    r_shuff_all = []
    for run_idx, (z_a, z_b) in enumerate(zip(z_a_runs, z_b_runs)):
        run_result = compute_geometry_preservation_run(z_a, z_b, n_shuff=n_shuff, rng=rng)
        r_real[run_idx] = run_result["rReal"]
        r_shuff[run_idx] = run_result["rShuff"]
        r_shuff_all.append(run_result["rShuffAll"])

    stats = paired_geometry_stats(r_real, r_shuff, n_permutations=n_permutations, rng=rng)
    if plot_path is not None:
        plot_paired_geometry_scores(r_real, r_shuff, plot_path, title_suffix=title_suffix)

    return {
        "rReal": r_real,
        "rShuff": r_shuff,
        "rShuffAll": r_shuff_all,
        "diff": r_real - r_shuff,
        "stats": stats,
        "plot_path": plot_path,
    }


def shuffle_geometry_scores(z_a, z_b, n_shuffles, rng):
    scores = np.zeros(n_shuffles)
    for shuffle_idx in range(n_shuffles):
        scores[shuffle_idx] = compute_geometry_preservation_run(z_a, z_b, n_shuff=1, rng=rng)["rShuff"]
    return scores


def plot_paired_geometry_scores(real_scores, shuff_scores, output_path, rat_id=None, title_suffix=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_scores = np.asarray(real_scores, dtype=float)
    shuff_scores = np.asarray(shuff_scores, dtype=float)
    stats = paired_geometry_stats(real_scores, shuff_scores)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for run_idx in range(len(real_scores)):
        ax.plot([0, 1], [shuff_scores[run_idx], real_scores[run_idx]], color="#9ca3af", linewidth=0.9, alpha=0.8)
    ax.scatter(np.zeros_like(shuff_scores), shuff_scores, s=35, alpha=0.9, color="#6b7280", label="Shuffled")
    ax.scatter(np.ones_like(real_scores), real_scores, s=35, alpha=0.9, color="#c2410c", label="Real")

    means = [stats["shuff_mean"], stats["real_mean"]]
    errors = [stats["shuff_sem"], stats["real_sem"]]
    ax.errorbar([0, 1], means, yerr=errors, color="#111827", marker="_", markersize=22, capsize=4, linewidth=1.4)

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Shuffled", "Real"])
    ax.set_ylabel("Spearman geometry-preservation score")
    label = f"Rat {rat_id}" if rat_id else "Rat/session"
    if title_suffix:
        label = f"{label} {title_suffix}"
    ax.set_title(label)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_single_rat(real_scores, shuffle_scores, output_path, rat_id=None, title_suffix=None):
    return plot_paired_geometry_scores(real_scores, shuffle_scores, output_path, rat_id=rat_id, title_suffix=title_suffix)


def run_geometry_preservation(
    traceA1An_An,
    traceAnB1_An,
    traceA1An_A1,
    traceAnB1_B1,
    CSUSAn,
    CSUSA1,
    CSUSB1,
    dimensions,
    iterations,
    parameter_set,
    parameter_set_name="unknown",
    shuffles=1,
    output_dimension=3,
    output_dir="geometry_preservation_outputs",
    rat_id=None,
    session_id=None,
    random_seed=None,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    if len(CSUSAn) % 10 == 9:
        CSUSAn = CSUSAn[9:]
        traceA1An_An = traceA1An_An[9:]
        traceAnB1_An = traceAnB1_An[9:]
    traceA1An_A1, CSUSA1 = trim_mod10_if_needed(traceA1An_A1, CSUSA1)
    traceAnB1_B1, CSUSB1 = trim_mod10_if_needed(traceAnB1_B1, CSUSB1)

    bins = common_bins(CSUSA1, CSUSB1)
    real_scores = np.zeros(iterations)
    shuffle_scores = np.zeros(iterations)
    shuffle_scores_all = np.zeros((iterations, shuffles))
    summary_rows = []
    shuffle_rows = []

    for run_idx in range(iterations):
        print(f"Geometry run {run_idx + 1}/{iterations}")

        model_a = make_cebra_model(parameter_set, output_dimension)
        model_b = make_cebra_model(parameter_set, output_dimension)

        model_a.fit(traceA1An_An, CSUSAn)
        model_b.fit(traceAnB1_An, CSUSAn)

        embedding_a = model_a.transform(traceA1An_A1)
        embedding_b = model_b.transform(traceAnB1_B1)

        z_a = bin_mean_embedding(embedding_a, CSUSA1, bins)
        z_b = bin_mean_embedding(embedding_b, CSUSB1, bins)

        run_geometry = compute_geometry_preservation_run(z_a, z_b, n_shuff=shuffles, rng=rng)
        real_score = run_geometry["rReal"]
        shuff_score = run_geometry["rShuff"]
        shuff_scores_all = run_geometry["rShuffAll"]
        real_scores[run_idx] = real_score
        shuffle_scores[run_idx] = shuff_score
        shuffle_scores_all[run_idx] = shuff_scores_all

        summary_rows.append(
            {
                "rat_id": rat_id,
                "session_id": session_id,
                "parameter_set_name": parameter_set_name,
                "dimensions_argument": dimensions,
                "output_dimension": output_dimension,
                "model_run": run_idx,
                "n_bins": len(bins),
                "n_shuff": shuffles,
                "rReal": real_score,
                "rShuff": shuff_score,
                "rDiff": real_score - shuff_score,
                "real_score": real_score,
                "shuffle_score": shuff_score,
            }
        )
        for shuffle_idx, shuffle_score in enumerate(shuff_scores_all):
            shuffle_rows.append(
                {
                    "rat_id": rat_id,
                    "session_id": session_id,
                    "parameter_set_name": parameter_set_name,
                    "dimensions_argument": dimensions,
                    "output_dimension": output_dimension,
                    "model_run": run_idx,
                    "shuffle_id": shuffle_idx,
                    "shuffle_score": shuffle_score,
                }
            )

    stats = paired_geometry_stats(real_scores, shuffle_scores, rng=rng)
    stats_rows = [{"rat_id": rat_id, "session_id": session_id, "parameter_set_name": parameter_set_name, **stats}]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rat_part = f"_{rat_id}" if rat_id else ""
    base = (
        f"geometry_preservation{rat_part}_{parameter_set_name}"
        f"_dim{output_dimension}_bins{dimensions}_{timestamp}"
    )
    summary_path = os.path.join(output_dir, f"{base}_summary.csv")
    shuffle_path = os.path.join(output_dir, f"{base}_shuffles.csv")
    stats_path = os.path.join(output_dir, f"{base}_stats.csv")
    npz_path = os.path.join(output_dir, f"{base}.npz")
    plot_path = os.path.join(output_dir, f"{base}.png")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(shuffle_rows).to_csv(shuffle_path, index=False)
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    np.savez(
        npz_path,
        rReal=real_scores,
        rShuff=shuffle_scores,
        rShuffAll=shuffle_scores_all,
        bins=bins,
        parameter_set_name=parameter_set_name,
        output_dimension=output_dimension,
        dimensions=dimensions,
        rat_id=rat_id if rat_id else "",
        session_id=session_id if session_id else "",
    )
    try:
        plot_paired_geometry_scores(real_scores, shuffle_scores, plot_path, rat_id=rat_id)
    except ImportError as exc:
        plot_path = None
        print(f"Skipping rat plot because matplotlib is unavailable: {exc}")

    print(f"Summary saved to {summary_path}")
    print(f"Shuffle distribution saved to {shuffle_path}")
    print(f"Run-level paired stats saved to {stats_path}")
    print(f"Raw arrays saved to {npz_path}")
    if plot_path:
        print(f"Rat plot saved to {plot_path}")
    print("Run-level paired stats:")
    print(f"  n_runs: {stats['n_runs']}")
    print(f"  rReal mean +/- SEM: {stats['real_mean']:.6f} +/- {stats['real_sem']:.6f}")
    print(f"  rShuff mean +/- SEM: {stats['shuff_mean']:.6f} +/- {stats['shuff_sem']:.6f}")
    print(f"  rReal - rShuff mean +/- SEM: {stats['diff_mean']:.6f} +/- {stats['diff_sem']:.6f}")
    print(
        "  paired sign-flip p(two-sided): "
        f"{stats['sign_flip_p_two_sided']:.6g} "
        f"(n_perm={stats['sign_flip_n_permutations']}, exact={stats['sign_flip_exact']})"
    )
    print(f"  paired t-test: t={stats['paired_t_stat']:.6f}, p(two-sided)={stats['paired_t_p_two_sided']:.6g}")

    return {
        "summary_path": summary_path,
        "shuffle_path": shuffle_path,
        "stats_path": stats_path,
        "npz_path": npz_path,
        "plot_path": plot_path,
        "rReal": real_scores,
        "rShuff": shuffle_scores,
        "stats": stats,
    }
