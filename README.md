# Hannahs-CEBRAs

Code and analysis outputs for CEBRA latent-space decoding and geometry comparisons in eyeblink-task data.



## Additional Analysis Script

### `cross_rat_latent_decoder.py`

Across-animal CEBRA latent-space task-epoch decoding.

The script asks whether a decoder trained on one rat's CEBRA latent space can decode task epochs in another rat after aligning the test rat's latent coordinates into the train rat's latent space. It does not match neurons across animals. Each rat keeps its own embedding, and transfer is evaluated after held-out Procrustes alignment.

Key behavior:

- Loads `.mat` files containing embeddings, task labels, and optional trial IDs.
- Tests ordered train/test rat pairs, including within-rat diagonal controls.
- Supports `CSUS2`, `CSUS5`, or other task-label schemes if the `.mat` files contain matching label keys.
- Tests one or more latent dimensionalities by taking the first `dim` embedding columns.
- Uses trial-level held-out alignment/decoding splits when `trial_ids` are present.
- Falls back to sample-level splits when trial IDs are absent, and records that warning in output rows.
- Fits a Procrustes transform from task-bin means on alignment data only.
- Trains a decoder on held-out train-rat samples and scores it on held-out aligned test-rat samples.
- Generates train-label shuffle controls.
- Writes CSV, NPZ, MAT, and SVG summary outputs.

Basic run:

```bash
python cross_rat_latent_decoder.py \
  --input_dir /path/to/mat_files \
  --output_dir cross_rat_decoder_outputs \
  --task_schemes CSUS2,CSUS5 \
  --dims 2,3,5,7,10 \
  --n_splits 20 \
  --n_shuffles 100 \
  --decoder logreg
```

Fast smoke test:

```bash
python cross_rat_latent_decoder.py \
  --input_dir /path/to/mat_files \
  --output_dir cross_rat_decoder_smoke_test \
  --task_schemes CSUS5 \
  --dims 3 \
  --n_splits 2 \
  --n_shuffles 5 \
  --decoder lda \
  --no_mat
```

Use custom `.mat` keys:

```bash
python cross_rat_latent_decoder.py \
  --input_dir /path/to/mat_files \
  --output_dir cross_rat_decoder_outputs \
  --embedding_key embedding \
  --label_key labels \
  --trial_key trial_ids \
  --file_pattern "*rat*.mat"
```

Decoder choices:

- `logreg`: balanced multinomial-style logistic regression, the default.
- `lda`: linear discriminant analysis.
- `knn`: 5-nearest-neighbor classifier.

Main outputs from `cross_rat_latent_decoder.py`:

- `cross_rat_decoding_results.csv`: one row per real or shuffle split for each ordered rat pair.
- `cross_rat_decoding_summary.csv`: per-pair means plus off-diagonal aggregate statistics.
- `confusion_matrices.npz`: compressed confusion matrices keyed by `confusion_matrix_key`.
- `cross_rat_decoding_results.mat`: MATLAB-compatible table export, unless `--no_mat` is used.
- `heatmap_real_accuracy_<TASK>_dim<DIM>.svg`: train-rat by test-rat real accuracy heatmap.
- `heatmap_shuffle_accuracy_<TASK>_dim<DIM>.svg`: matching shuffle heatmap.
- `summary_bar_<TASK>_dim<DIM>.svg`: across-rat real, shuffle, and within-rat control summary.

Read the summary in Python:

```python
import pandas as pd

summary = pd.read_csv("cross_rat_decoder_outputs/cross_rat_decoding_summary.csv")
off_diag = summary[summary["row_type"] == "aggregate_off_diagonal"]
print(off_diag[[
    "task_scheme",
    "dim",
    "real_accuracy_mean",
    "shuffle_accuracy_mean",
    "accuracy_effect_real_minus_shuffle",
    "paired_t_p_value",
    "wilcoxon_p_value",
]])
```

Load confusion matrices:

```python
import numpy as np
import pandas as pd

results = pd.read_csv("cross_rat_decoder_outputs/cross_rat_decoding_results.csv")
confusions = np.load("cross_rat_decoder_outputs/confusion_matrices.npz")

first_key = results.loc[results["performance_type"] == "real", "confusion_matrix_key"].iloc[0]
print(first_key)
print(confusions[first_key])
```

## Added Geometry Output Bundle

### `geometry/`

This directory collects geometry and Procrustes output tables/figures that were moved under a single namespace. The root-level folders with the same names are currently deleted in the working tree, so the active location is `geometry/...`.

The tables are analysis outputs, not source code. They can be read directly with pandas, MATLAB, R, or spreadsheet software.

```python
import pandas as pd

stats = pd.read_csv(
    "geometry/cross_rat_procrustes_shuffle_null_outputs/"
    "cross_rat_procrustes_shuffle_null_stats.csv"
)
print(stats[[
    "comparison_family",
    "actual_mean_procrustes_disparity",
    "shuffle_null_mean_procrustes_disparity",
    "one_sided_p_lower_than_shuffle_procrustes_disparity",
]])
```

### `geometry/geometry_matrix_stats_outputs/`

Per-rat matrix RSA shuffle-null statistics for the geometry-preservation outputs.

Files:

- `per_rat_matrix_rsa_shuffle_null_plot_data.csv`: long-form real and shuffle scores for plotting.
- `per_rat_matrix_rsa_shuffle_null_stats.csv`: per-rat summary statistics, confidence intervals, shuffle-null percentiles, and one-sided permutation p-values.

Example:

```python
import pandas as pd

matrix_stats = pd.read_csv(
    "geometry/geometry_matrix_stats_outputs/per_rat_matrix_rsa_shuffle_null_stats.csv"
)
print(matrix_stats[["rat_id", "mean_rReal", "shuffle_null_mean", "one_sided_permutation_p"]])
```

### `geometry/procrustes_shuffle_null_outputs/`

Within-rat Procrustes shuffle-null statistics comparing task-bin geometry across environments/runs.

Files:

- `within_rat_procrustes_shuffle_null_plot_data.csv`: long-form real and shuffle Procrustes/RMSE scores.
- `within_rat_procrustes_shuffle_null_stats.csv`: per-rat real means, SEMs, confidence intervals, shuffle-null percentiles, and p-values.

Example:

```python
import pandas as pd

within = pd.read_csv(
    "geometry/procrustes_shuffle_null_outputs/within_rat_procrustes_shuffle_null_stats.csv"
)
print(within[[
    "rat_id",
    "mean_real_procrustes_disparity",
    "shuffle_null_mean_procrustes_disparity",
    "one_sided_p_lower_than_shuffle_procrustes_disparity",
]])
```

### `geometry/cross_rat_geometry_shuffle_null_outputs/`

Cross-rat geometry-preservation shuffle-null results based on correlation-style geometry scores.

Files:

- `cross_rat_geometry_run_scores.csv`: per model-run, per rat-pair real scores.
- `cross_rat_geometry_actual_rat_pair_means.csv`: real score means for each rat pair.
- `cross_rat_geometry_shuffle_null_means.csv`: shuffle-null mean score samples.
- `cross_rat_geometry_shuffle_null_plot_data.csv`: combined actual and null values for plotting.
- `cross_rat_geometry_shuffle_null_stats.csv`: comparison-family summary statistics and one-sided permutation p-values.

Example:

```python
import pandas as pd

cross_geom = pd.read_csv(
    "geometry/cross_rat_geometry_shuffle_null_outputs/"
    "cross_rat_geometry_shuffle_null_stats.csv"
)
print(cross_geom[[
    "comparison_family",
    "actual_mean",
    "shuffle_null_mean",
    "actual_minus_shuffle_null_mean",
    "one_sided_permutation_p",
]])
```

### `geometry/cross_rat_procrustes_shuffle_null_outputs/`

Cross-rat Procrustes shuffle-null results. Lower Procrustes disparity, aligned RMSE, and SSE-over-target-SS indicate better alignment.

Files:

- `cross_rat_procrustes_run_scores.csv`: per model-run, per rat-pair Procrustes/RMSE scores.
- `cross_rat_procrustes_actual_rat_pair_means.csv`: real pair means.
- `cross_rat_procrustes_shuffle_null_means.csv`: shuffle-null mean samples.
- `cross_rat_procrustes_shuffle_null_plot_data.csv`: combined actual and null values for plotting.
- `cross_rat_procrustes_shuffle_null_stats.csv`: comparison-family statistics and lower-than-null one-sided p-values.

Example:

```python
import pandas as pd

cross_proc = pd.read_csv(
    "geometry/cross_rat_procrustes_shuffle_null_outputs/"
    "cross_rat_procrustes_shuffle_null_stats.csv"
)
print(cross_proc[[
    "comparison_family",
    "actual_mean_procrustes_disparity",
    "shuffle_null_mean_procrustes_disparity",
    "actual_minus_shuffle_null_mean_procrustes_disparity",
    "one_sided_p_lower_than_shuffle_procrustes_disparity",
]])
```

### `geometry/procrustes_disparity_outputs/within_rat/`

Within-rat A/B Procrustes alignment outputs.

Files:

- `within_rat_procrustes_aligned_2026-05-08_13-17-14.csv`: aligned coordinates for each rat, environment, and task bin.
- `within_rat_procrustes_disparity_2026-05-08_13-17-14.csv`: per-rat Procrustes disparity and RMSE metrics.
- `within_rat_procrustes_2d_rat0222_2026-05-08_13-17-14.pdf`: 2D plot for rat0222.
- `within_rat_procrustes_2d_rat0307_2026-05-08_13-17-14.pdf`: 2D plot for rat0307.
- `within_rat_procrustes_2d_rat0313_2026-05-08_13-17-14.pdf`: 2D plot for rat0313.
- `within_rat_procrustes_2d_rat0314_2026-05-08_13-17-14.pdf`: 2D plot for rat0314.
- `within_rat_procrustes_2d_rat0816_2026-05-08_13-17-14.pdf`: 2D plot for rat0816.
- Matching `.png` and `.svg` versions of these plots are present locally but ignored by git because `.png` and `.svg` are ignored in `.gitignore`.
- `within_rat_procrustes_2d_combined_2026-05-08_13-17-14.png` and `.svg`: combined local plot files, also ignored by git.

Example:

```python
import pandas as pd

aligned = pd.read_csv(
    "geometry/procrustes_disparity_outputs/within_rat/"
    "within_rat_procrustes_aligned_2026-05-08_13-17-14.csv"
)
rat0314 = aligned[aligned["rat_id"] == "rat0314"]
print(rat0314[["environment", "task_bin", "x", "y", "z"]])
```

### `geometry/procrustes_disparity_outputs/cross_rat_mean_ab/`

Cross-rat Procrustes alignment of each rat's mean A/B task-bin geometry into a shared view.

Files:

- `cross_rat_mean_ab_procrustes_2026-05-08_13-17-30.csv`: aligned cross-rat coordinates.
- `cross_rat_mean_ab_procrustes_disparity_2026-05-08_13-17-30.csv`: disparity/RMSE metrics for the alignment.
- `cross_rat_mean_ab_procrustes_pca2d_scores_2026-05-08_13-17-30.csv`: aligned coordinates plus PC1/PC2 scores.
- `cross_rat_mean_ab_procrustes_pca2d_info_2026-05-08_13-17-30.csv`: PCA explained variance information.
- `cross_rat_mean_ab_procrustes_pca2d_2026-05-08_13-17-30.pdf`: 2D PCA figure.
- `cross_rat_mean_ab_procrustes_group_mean_3d_2026-05-08_13-17-30.pdf`: 3D group-mean figure.
- Matching `.png` and `.svg` versions are present locally but ignored by git.

Example:

```python
import pandas as pd

pca_scores = pd.read_csv(
    "geometry/procrustes_disparity_outputs/cross_rat_mean_ab/"
    "cross_rat_mean_ab_procrustes_pca2d_scores_2026-05-08_13-17-30.csv"
)
print(pca_scores.groupby("rat_id")[["PC1", "PC2"]].mean())
```

### `geometry/procrustes_disparity_outputs/aligned_cross_rat/`

Aligned cross-rat embedding-similarity coordinates and disparity metrics.

Files:

- `aligned_cross_rat_embeddings_similarity_2026-05-08_13-17-41.csv`: aligned coordinates by rat, environment, and task bin.
- `aligned_cross_rat_embeddings_similarity_disparity_2026-05-08_13-17-41.csv`: alignment method, reference rat/environment, disparity, RMSE, and SSE metrics.
- Matching `.png` and `.svg` plot files are present locally but ignored by git.

Example:

```python
import pandas as pd

similarity = pd.read_csv(
    "geometry/procrustes_disparity_outputs/aligned_cross_rat/"
    "aligned_cross_rat_embeddings_similarity_disparity_2026-05-08_13-17-41.csv"
)
print(similarity[[
    "rat_id",
    "environment",
    "reference_rat_id",
    "procrustes_disparity",
    "aligned_rmse",
]])
```

### `geometry/procrustes_disparity_ad_hoc_outputs/.DS_Store`

macOS metadata file. It is not part of the analysis and can be ignored.

## Notes On Current Working Tree

The current working tree shows `geometry/` and `cross_rat_latent_decoder.py` as new files, with old root-level output directories marked deleted. That is consistent with the outputs being reorganized under `geometry/`.

To inspect the current added/untracked files:

```bash
git status --short
git ls-files --others --exclude-standard
```

To list local ignored plot images that still exist under `geometry/`:

```bash
git status --short --ignored geometry
```
