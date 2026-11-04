# Position decoding corrections

All position shuffle paths now permute complete training rows after the split.
Held-out and external test positions retain their true order. This preserves
training occupancy and the joint x/y distribution, without importing positions
from held-out frames into the shuffled training set.

## Metrics and compatibility

`pos_score` retains its four-value return order:

1. Uniform-average coordinate R².
2. Median of pooled absolute coordinate errors (legacy `KNN_err_*` / `Knn_pos_err`).
3. Mean Euclidean distance per frame.
4. **Median Euclidean distance per frame — primary spatial error.**

Errors use the original coordinate units. The grid's `mead_test` remains as a
legacy alias; `median_distance_test` is the explicit new name. `shuf_med` is
shuffled minus real median distance, so positive values favor the real decoder.
The dimension-grid CSV header now has one `output_dimension` column, matching
its 33 data columns.

The decoder explicitly uses continuous kNN regression, even for integer-valued
coordinates. Default k remains 36; legacy notebook wrappers retain k=32 and
cosine decoder distance. Invalid shapes, nonfinite inputs, mismatched sample
counts, and insufficient training samples raise clear errors.

Averaging shuffled neighbors can predict near the training center. This is
expected and is not fixed by changing k only for the null. Grids additionally
report `center_median_distance_test`, computed from a constant mean of training
positions, and `median_distance_shuffle` to make this behavior visible.

## Two existing null designs

- `pos_compare_iterations*.py` retains its **decoder-only** null: CEBRA learns
  from real training positions, then kNN is fitted to shuffled training labels
  on the same embeddings. Test labels are now always true positions.
- Grid scripts, `pos_compare.py`, `pos_decoding_self.py`, and the deprecated
  A-vs-B decoder use a **training-label pipeline** null: both CEBRA and kNN
  receive the same shuffled training positions. The older self/A-vs-B paths
  previously fitted kNN to real labels despite shuffled CEBRA supervision.

These nulls answer different questions; do not pool them as one control.
Notebook exports now share the scorer and use joint-row shuffles. The A22
export also uses matching training/test splits and the shuffled model for both
shuffled embeddings. These exports still require their original notebook data
and plotting context.

## Distance and historical results

Both grids accept `--distance cosine|euclidean` (default cosine for compatibility).
All position SLURM launchers now specify their intended distance explicitly.
Previously the launchers named `euc`, including 0313, invoked grid2 with cosine
hardcoded. Grid2 retains constant temperature; grid1 retains automatic temperature.
kNN distance remains Euclidean independently of CEBRA's training distance.

Existing results have not been regenerated or rewritten. Rerun affected analyses
before comparing corrected shuffle controls. Final changes were not runtime
validated: the test attempt was stopped and no further tests were run at the
user's request.
