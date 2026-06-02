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


def trim_mod10_with_optional_ids(traces, labels, trial_ids=None):
    min_length = len(labels)
    if min_length % 10 == 9:
        traces = traces[9:]
        labels = labels[9:]
        if trial_ids is not None:
            trial_ids = np.asarray(trial_ids)[9:]
    return traces, labels, trial_ids


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

    if n_shuff < 0:
        raise ValueError("n_shuff must be non-negative.")

    r_real = geometry_preservation_score(z_a, z_b)
    r_shuff_all = np.zeros(n_shuff)
    for shuffle_idx in range(n_shuff):
        permutation = rng.permutation(z_b.shape[0])
        r_shuff_all[shuffle_idx] = geometry_preservation_score(z_a, z_b[permutation])
    r_shuff = np.nan if n_shuff == 0 else np.nanmean(r_shuff_all)

    return {
        "rReal": r_real,
        "rShuff": r_shuff,
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


def extract_cebra_loss_history(model):
    """Return a CEBRA training-loss history if this CEBRA version exposes one."""
    state = getattr(model, "state_dict_", None)
    if isinstance(state, dict) and "loss" in state:
        loss = np.asarray(state["loss"], dtype=float).reshape(-1)
        return loss
    for attr in ("loss_", "loss_history_", "training_loss_", "history_"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, dict) and "loss" in value:
                value = value["loss"]
            try:
                return np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
    return np.array([], dtype=float)


def final_loss_from_history(loss_history):
    loss_history = np.asarray(loss_history, dtype=float).reshape(-1)
    finite = loss_history[np.isfinite(loss_history)]
    return float(finite[-1]) if finite.size else np.nan


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
    CSUSA1_trial_ids=None,
    CSUSB1_trial_ids=None,
    save_branch="both",
):
    comparison_mode = "An_vs_B1_separately_trained"
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    save_branch = str(save_branch).lower()
    if save_branch not in {"both", "a", "b"}:
        raise ValueError("save_branch must be one of: both, A, B.")
    save_a_branch = save_branch in {"both", "a"}
    save_b_branch = save_branch in {"both", "b"}
    compute_ab_geometry = save_branch == "both"

    if len(CSUSAn) % 10 == 9:
        CSUSAn = CSUSAn[9:]
        traceA1An_An = traceA1An_An[9:]
        traceAnB1_An = traceAnB1_An[9:]
    traceA1An_A1, CSUSA1, CSUSA1_trial_ids = trim_mod10_with_optional_ids(traceA1An_A1, CSUSA1, CSUSA1_trial_ids)
    traceAnB1_B1, CSUSB1, CSUSB1_trial_ids = trim_mod10_with_optional_ids(traceAnB1_B1, CSUSB1, CSUSB1_trial_ids)
    if CSUSA1_trial_ids is not None and len(CSUSA1_trial_ids) != len(CSUSA1):
        raise ValueError("CSUSA1_trial_ids must match CSUSA1 length after trimming/filtering.")
    if CSUSB1_trial_ids is not None and len(CSUSB1_trial_ids) != len(CSUSB1):
        raise ValueError("CSUSB1_trial_ids must match CSUSB1 length after trimming/filtering.")

    if compute_ab_geometry:
        bins = common_bins(CSUSAn, CSUSB1)
    elif save_a_branch:
        bins = common_bins(CSUSAn, CSUSAn)
    else:
        bins = common_bins(CSUSB1, CSUSAn)
    real_scores = np.full(iterations, np.nan)
    shuffle_scores = np.full(iterations, np.nan)
    shuffle_scores_all = np.zeros((iterations, shuffles))
    z_a_runs = None
    z_b_runs = None
    z_an_a_runs = None
    z_an_b_runs = None
    embedding_a_runs = None
    embedding_b_runs = None
    embedding_an_a_runs = None
    embedding_an_b_runs = None
    loss_a_runs = np.full(iterations, np.nan)
    loss_b_runs = np.full(iterations, np.nan)
    loss_a_history_runs = []
    loss_b_history_runs = []
    summary_rows = []
    shuffle_rows = []

    for run_idx in range(iterations):
        print(f"Geometry run {run_idx + 1}/{iterations}")

        z_a = None
        z_b = None
        if save_a_branch:
            model_a = make_cebra_model(parameter_set, output_dimension)
            model_a.fit(traceAnB1_An, CSUSAn)
            loss_a_history = extract_cebra_loss_history(model_a)
            loss_a_runs[run_idx] = final_loss_from_history(loss_a_history)
            loss_a_history_runs.append(loss_a_history)
            embedding_an_a = model_a.transform(traceAnB1_An)
            embedding_a = embedding_an_a
            z_an_a = bin_mean_embedding(embedding_an_a, CSUSAn, bins)
            z_a = z_an_a
            if z_a_runs is None:
                z_a_runs = np.zeros((iterations, z_a.shape[0], z_a.shape[1]))
                z_an_a_runs = np.zeros((iterations, z_an_a.shape[0], z_an_a.shape[1]))
                embedding_a_runs = np.zeros((iterations, embedding_a.shape[0], embedding_a.shape[1]))
                embedding_an_a_runs = np.zeros((iterations, embedding_an_a.shape[0], embedding_an_a.shape[1]))
            z_a_runs[run_idx] = z_a
            z_an_a_runs[run_idx] = z_an_a
            embedding_a_runs[run_idx] = embedding_a
            embedding_an_a_runs[run_idx] = embedding_an_a

        if save_b_branch:
            model_b = make_cebra_model(parameter_set, output_dimension)
            model_b.fit(traceAnB1_B1, CSUSB1)
            loss_b_history = extract_cebra_loss_history(model_b)
            loss_b_runs[run_idx] = final_loss_from_history(loss_b_history)
            loss_b_history_runs.append(loss_b_history)
            embedding_an_b = model_b.transform(traceAnB1_An)
            embedding_b = model_b.transform(traceAnB1_B1)
            z_an_b = bin_mean_embedding(embedding_an_b, CSUSAn, bins)
            z_b = bin_mean_embedding(embedding_b, CSUSB1, bins)
            if z_b_runs is None:
                z_b_runs = np.zeros((iterations, z_b.shape[0], z_b.shape[1]))
                z_an_b_runs = np.zeros((iterations, z_an_b.shape[0], z_an_b.shape[1]))
                embedding_b_runs = np.zeros((iterations, embedding_b.shape[0], embedding_b.shape[1]))
                embedding_an_b_runs = np.zeros((iterations, embedding_an_b.shape[0], embedding_an_b.shape[1]))
            z_b_runs[run_idx] = z_b
            z_an_b_runs[run_idx] = z_an_b
            embedding_b_runs[run_idx] = embedding_b
            embedding_an_b_runs[run_idx] = embedding_an_b

        run_shuff_scores = np.zeros(shuffles)
        if compute_ab_geometry:
            run_geometry = compute_geometry_preservation_run(z_a, z_b, n_shuff=shuffles, rng=rng)
            real_score = run_geometry["rReal"]
            shuff_score = run_geometry["rShuff"]
            run_shuff_scores = run_geometry["rShuffAll"]
            real_scores[run_idx] = real_score
            shuffle_scores[run_idx] = shuff_score
            shuffle_scores_all[run_idx] = run_shuff_scores
        else:
            real_score = np.nan
            shuff_score = np.nan

        summary_rows.append(
            {
                "rat_id": rat_id,
                "session_id": session_id,
                "parameter_set_name": parameter_set_name,
                "comparison_mode": comparison_mode,
                "dimensions_argument": dimensions,
                "output_dimension": output_dimension,
                "model_run": run_idx,
                "save_branch": save_branch,
                "n_bins": len(bins),
                "n_shuff": shuffles,
                "rReal": real_score,
                "rShuff": shuff_score,
                "rDiff": real_score - shuff_score,
                "real_score": real_score,
                "shuffle_score": shuff_score,
                "lossA": loss_a_runs[run_idx],
                "lossB": loss_b_runs[run_idx],
            }
        )
        for shuffle_idx, shuffle_score in enumerate(run_shuff_scores):
            shuffle_rows.append(
                {
                    "rat_id": rat_id,
                    "session_id": session_id,
                    "parameter_set_name": parameter_set_name,
                    "comparison_mode": comparison_mode,
                    "dimensions_argument": dimensions,
                    "output_dimension": output_dimension,
                    "model_run": run_idx,
                    "shuffle_id": shuffle_idx,
                    "shuffle_score": shuffle_score,
                }
            )

    stats = paired_geometry_stats(real_scores, shuffle_scores, rng=rng)
    if shuffles == 0 or not compute_ab_geometry:
        stats.update(
            {
                "n_runs": iterations,
                "real_mean": np.nan if not compute_ab_geometry else (np.nanmean(real_scores) if len(real_scores) else np.nan),
                "real_sem": np.nan if not compute_ab_geometry else sem(real_scores),
                "shuff_mean": np.nan,
                "shuff_sem": np.nan,
                "diff_mean": np.nan,
                "diff_sem": np.nan,
                "sign_flip_p_two_sided": np.nan,
                "sign_flip_n_permutations": 0,
                "sign_flip_exact": False,
                "paired_t_stat": np.nan,
                "paired_t_p_two_sided": np.nan,
            }
        )
    stats_rows = [
        {
            "rat_id": rat_id,
            "session_id": session_id,
            "parameter_set_name": parameter_set_name,
            "comparison_mode": comparison_mode,
            "save_branch": save_branch,
            **stats,
        }
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rat_part = f"_{rat_id}" if rat_id else ""
    branch_part = "" if save_branch == "both" else f"_{save_branch}branch"
    base = (
        f"geometry_preservation{rat_part}_{parameter_set_name}"
        f"_dim{output_dimension}_bins{dimensions}{branch_part}_{timestamp}"
    )
    summary_path = os.path.join(output_dir, f"{base}_summary.csv")
    shuffle_path = os.path.join(output_dir, f"{base}_shuffles.csv")
    stats_path = os.path.join(output_dir, f"{base}_stats.csv")
    npz_path = os.path.join(output_dir, f"{base}.npz")
    plot_path = os.path.join(output_dir, f"{base}.png")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(shuffle_rows).to_csv(shuffle_path, index=False)
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    save_dict = {
        "rReal": real_scores,
        "rShuff": shuffle_scores,
        "rShuffAll": shuffle_scores_all,
        "labelsAn": CSUSAn,
        "sample_indicesAn": np.arange(len(CSUSAn)),
        "trial_idsAn": np.array([]),
        "bins": bins,
        "parameter_set_name": parameter_set_name,
        "comparison_mode": comparison_mode,
        "output_dimension": output_dimension,
        "dimensions": dimensions,
        "rat_id": rat_id if rat_id else "",
        "session_id": session_id if session_id else "",
        "save_branch": save_branch,
    }
    if save_a_branch:
        save_dict.update(
            {
                "zA_runs": z_a_runs,
                "zAnA_runs": z_an_a_runs,
                "embeddingA_runs": embedding_a_runs,
                "embeddingAnA_runs": embedding_an_a_runs,
                "lossA_runs": loss_a_runs,
                "lossA_history_runs": np.asarray(loss_a_history_runs, dtype=object),
                "labelsA": CSUSAn,
                "sample_indicesA": np.arange(len(CSUSAn)),
                "trial_idsA": np.array([]),
            }
        )
    if save_b_branch:
        save_dict.update(
            {
                "zB_runs": z_b_runs,
                "zAnB_runs": z_an_b_runs,
                "embeddingB_runs": embedding_b_runs,
                "embeddingAnB_runs": embedding_an_b_runs,
                "lossB_runs": loss_b_runs,
                "lossB_history_runs": np.asarray(loss_b_history_runs, dtype=object),
                "labelsB": CSUSB1,
                "sample_indicesB": np.arange(len(CSUSB1)),
                "trial_idsB": CSUSB1_trial_ids if CSUSB1_trial_ids is not None else np.array([]),
            }
        )
    np.savez(npz_path, **save_dict)
    try:
        if shuffles == 0 or not compute_ab_geometry:
            plot_path = None
            reason = "--shuffles 0 was requested" if shuffles == 0 else f"--save_branch {save_branch} skips A-vs-B geometry"
            print(f"Skipping rat geometry shuffle plot because {reason}.")
        else:
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
    print(f"  n_runs: {iterations}")
    if not compute_ab_geometry:
        print(f"  A-vs-B geometry stats skipped (--save_branch {save_branch}).")
    else:
        print(f"  rReal mean +/- SEM: {stats['real_mean']:.6f} +/- {stats['real_sem']:.6f}")
    if shuffles == 0:
        print("  geometry shuffle controls skipped (--shuffles 0).")
    elif compute_ab_geometry:
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
