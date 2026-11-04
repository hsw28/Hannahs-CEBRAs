"""Shared continuous-position decoder and metrics.

The four-value return order is retained for existing CSV consumers. The fourth
value (median Euclidean distance per frame) is the primary spatial error.
"""

import numbers
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f"{name} must be a nonempty 2D array")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def shuffle_positions(label_train, rng=None):
    """Permute complete position rows AFTER splitting; never pass test labels.

    Preserves the joint coordinate distribution. Uses NumPy's global seed by
    default, or accepts a Generator/RandomState for reproducible local shuffles.
    """
    labels = _matrix(label_train, "label_train")
    rng = np.random if rng is None else rng
    return labels[rng.permutation(len(labels))].copy()


def position_metrics(label_test, prediction):
    """Return R², pooled coordinate median AE, mean distance, median distance.

    Distances are Euclidean per frame (2D for x/y inputs), in the input position
    units. Pooled coordinate median AE is retained only as a legacy metric; it
    is not a spatial distance. Test labels must always be the true positions.
    """
    labels = _matrix(label_test, "label_test")
    prediction = _matrix(prediction, "prediction")
    if labels.shape != prediction.shape:
        raise ValueError("prediction and label_test must have matching shapes")
    residual = prediction - labels
    distances = np.linalg.norm(residual, axis=1)
    score = r2_score(labels, prediction) if len(labels) >= 2 else float("nan")
    return (float(score), float(np.median(np.abs(residual))),
            float(np.mean(distances)), float(np.median(distances)))


def position_center_score(label_train, label_test):
    """Score a constant training-mean position, using no test information.

    This diagnostic exposes how well occupancy alone can explain low errors.
    A shuffled kNN regressor can approach this predictor by averaging labels;
    that is expected behavior, not a reason to change k for the shuffled arm.
    """
    train = _matrix(label_train, "label_train")
    test = _matrix(label_test, "label_test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("training and test positions must have matching dimensions")
    return position_metrics(test, np.broadcast_to(train.mean(axis=0), test.shape))


def pos_score(emb_train, emb_test, label_train, label_test, n_neighbors=36,
              metric="euclidean"):
    """Fit continuous kNN regression and score true test positions.

    Always regress, including integer-valued coordinates (CEBRA's automatic
    decoder chooses classification for integer labels). Use identical k and
    distance for real and shuffled fits. Reject undersized training sets rather
    than silently changing the requested decoder.
    """
    train = _matrix(emb_train, "emb_train")
    test = _matrix(emb_test, "emb_test")
    labels = _matrix(label_train, "label_train")
    targets = _matrix(label_test, "label_test")
    if len(train) != len(labels) or len(test) != len(targets):
        raise ValueError("embedding and position sample counts must match")
    if train.shape[1] != test.shape[1] or labels.shape[1] != targets.shape[1]:
        raise ValueError("training and test dimensions must match")
    if (isinstance(n_neighbors, bool) or not isinstance(n_neighbors, numbers.Integral)
            or not 1 <= n_neighbors <= len(train)):
        raise ValueError("n_neighbors must be an integer between 1 and the training sample count")
    decoder = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
    decoder.fit(train, labels)
    return position_metrics(targets, decoder.predict(test))
