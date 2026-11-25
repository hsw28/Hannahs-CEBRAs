"""Within-animal An/B1 position consistency with independently fitted controls."""
import csv
import gc
import json
import random
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parameter_sets():
    rats = ['0222', '0307', '0313', '0314', '0816']
    groups = [
        [(5e-7, 1e-9, 60000, 'euclidean'), (5e-7, 1e-9, 120000, 'cosine'),
         (7.5e-5, 1.5, 200000, 'cosine'), (2e-4, 1e-9, 200000, 'cosine'),
         (7.5e-4, 1e-9, 240000, 'cosine')],
        [(5.5e-5, 1e-9, 25000, 'cosine'), (0.000622, 1.33, 21000, 'cosine'),
         (0.00055, 0.95, 26500, 'cosine'), (0.001, 1e-9, 30000, 'cosine'),
         (0.001, 0.15, 18000, 'cosine')],
    ]
    return {str(i + 1): {'set' + rat: dict(learning_rate=lr, min_temperature=temp,
            max_iterations=n, distance=distance, temperature_mode='auto')
            for rat, (lr, temp, n, distance) in zip(rats, group)}
            for i, group in enumerate(groups)}


def prepare_session(trace, position, prepared=False, min_speed=4):
    """Default MATLAB convention: cells x time; position rows = time,x,y at 2x rate."""
    from smoothpos import smoothpos
    from ca_velocity import ca_velocity
    trace, position = np.asarray(trace), np.asarray(position)
    if trace.ndim != 2 or position.ndim != 2:
        raise ValueError('Trace and position must be two-dimensional arrays.')
    if prepared:
        if position.shape[1] != 2:
            raise ValueError('Prepared positions must be samples x 2 (x,y).')
    else:
        trace = trace.T
        if position.shape[1] != 3:
            raise ValueError('Raw positions must have columns time,x,y.')
        if not np.isfinite(position).all():
            raise ValueError('Raw positions contain nonfinite values; clean before smoothing.')
        position = smoothpos(position.astype(float))[:, 1:][::2]
        if len(position) < len(trace):
            raise ValueError('Downsampled position is shorter than neural data.')
        position = position[:len(trace)]
        # Assign inter-frame speed to the ending frame, consistently at boundaries.
        keep = np.flatnonzero(ca_velocity(position) >= min_speed) + 1
        trace, position = trace[keep], position[keep]
    if len(trace) != len(position) or len(trace) < 20:
        raise ValueError('Need matching neural/position lengths and at least 20 samples.')
    if not np.isfinite(trace).all() or not np.isfinite(position).all():
        raise ValueError('Neural data and positions must be finite.')
    return np.asarray(trace, dtype=np.float32), np.asarray(position, dtype=np.float32)


def match_positions(a, b, max_distance, min_pairs):
    """Mutual nearest x/y samples: one-to-one, no latent centroids or interpolation."""
    distance, b_index = cKDTree(b).query(a)
    _, a_index = cKDTree(a).query(b)
    ia = np.flatnonzero((a_index[b_index] == np.arange(len(a))) & (distance <= max_distance))
    if len(ia) < min_pairs:
        raise ValueError('Only %d mutual position matches; need %d. Check coordinate registration, '
                         'spatial overlap and --max-match-distance.' % (len(ia), min_pairs))
    return ia, b_index[ia], distance[ia]


def fit_best(x, y, config, repeats, seed, path, name, shuffled):
    import torch
    from cebra import CEBRA
    best_loss, best_embedding, records = np.inf, None, []
    # One fixed permutation per session/control, matching the conditioning procedure.
    permutation = np.random.default_rng(seed).permutation(len(x)) if shuffled else np.arange(len(x))
    training_x = x[permutation]
    np.save(path / (name + '_training_permutation.npy'), permutation)
    runs_path = path / 'all_runs' / name
    runs_path.mkdir(parents=True, exist_ok=False)
    for run in range(repeats):
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        model = CEBRA(**config)  # Fresh instance: never retain references to a refitted model.
        model.fit(training_x, y)
        loss = float(model.state_dict_['loss'][-1])
        if not np.isfinite(loss):
            raise ValueError('%s run %d produced nonfinite loss.' % (name, run))
        records.append(dict(model=name, run=run, seed=run_seed, final_loss=loss))
        # Save every fit before selection; rows correspond to original neural data
        # and positions, including for models trained on shuffled neural rows.
        embedding = model.transform(x)
        np.savez_compressed(runs_path / ('run_%03d.npz' % run),
                            embedding=embedding, position=y, run=run,
                            seed=run_seed, final_loss=loss, model=name,
                            shuffled=shuffled)
        if loss < best_loss:
            best_loss = loss
            model.save(str(path / (name + '.pt')))
            # Transform original neural rows, as in the conditioning shuffled-model control.
            best_embedding = embedding
            (path / (name + '_selected.json')).write_text(json.dumps(records[-1], indent=2))
        print('%s run %d/%d: loss %.6g; best %.6g' % (name, run + 1, repeats, loss, best_loss), flush=True)
        del model
        gc.collect()
    return best_embedding, records


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(sessions, config, output, repeats=20, seed=42, max_distance=2., min_pairs=100,
                 metadata=None):
    from cebra.integrations.sklearn.metrics import consistency_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ia, ib, distance = match_positions(sessions[0][1], sessions[1][1], max_distance,
                                      max(min_pairs, config['output_dimension'] + 2))
    manifest = dict(config=config, repeats=repeats, seed=seed, max_match_distance=max_distance,
                    matched_samples=len(ia), matching='mutual nearest x/y; common coordinate system required',
                    score='CEBRA linear regression R2 on position-matched individual samples; in-sample',
                    control='fixed neural-row permutation for training; transform original neural rows',
                    all_run_embeddings='all_runs/{model}/run_{run:03d}.npz; zero-based run indices',
                    geometry_pairing='Pair An and B1 by run index, likewise shuffled controls; independent seeds',
                    metadata=metadata)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(output / 'position_matches.npz', index_An=ia, index_B1=ib,
                        distance=distance, position_An=sessions[0][1][ia], position_B1=sessions[1][1][ib])
    names = ['An', 'B1', 'An_shuffled', 'B1_shuffled']
    aligned, losses = [], []
    for k, name in enumerate(names):
        session = k % 2
        x, y = sessions[session]
        embedding, records = fit_best(x, y, config, repeats, seed + k * repeats,
                                       output, name, k >= 2)
        np.savez_compressed(output / (name + '_embedding.npz'), embedding=embedding, position=y)
        aligned.append(embedding[ia if session == 0 else ib])
        losses.extend(records)
        write_csv(output / 'training_losses.csv', losses, ['model', 'run', 'seed', 'final_loss'])
    # Correspondence is established above. "runs" bypasses CEBRA's 1D-only label alignment.
    scores, pairs, _ = consistency_score(embeddings=aligned, between='runs')
    matrix = np.full((4, 4), np.nan)
    rows = []
    for score, (source, target) in zip(scores, pairs):
        source, target = int(source), int(target)
        matrix[source, target] = score
        rows.append(dict(source=names[source], target=names[target], r2=float(score),
                         percent=float(score * 100), n_matches=len(ia)))
    write_csv(output / 'consistency_pairs.csv', rows, ['source', 'target', 'r2', 'percent', 'n_matches'])
    write_csv(output / 'consistency_matrix.csv', [dict(source=name, **dict(zip(names, row)))
              for name, row in zip(names, matrix)], ['source'] + names)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix * 100, vmin=0, vmax=100, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(names, rotation=35, ha='right'); ax.set_yticklabels(names)
    ax.set_xlabel('Target embedding'); ax.set_ylabel('Source embedding')
    ax.set_title('Within-animal position consistency: An vs B1')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '—' if i == j else '%.1f' % (100 * matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > .6 else 'black')
    fig.colorbar(image, ax=ax, label='Consistency (R² × 100)')
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(output / ('consistency_heatmap.' + extension), dpi=180)
    plt.close(fig)
    return matrix
