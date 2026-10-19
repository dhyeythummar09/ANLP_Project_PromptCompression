"""Paired inference with base-question cluster resampling."""
import numpy as np

def cluster_ci(values, groups, bootstrap=1000, seed=42):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("Finite values and one group per observation required")
    unique = np.unique(groups)
    if not len(unique):
        return (float("nan"), float("nan"))
    if len(unique) < 2:
        return (float("nan"), float("nan"))
    sums = np.array([values[groups == group].sum() for group in unique])
    counts = np.array([(groups == group).sum() for group in unique])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(bootstrap):
        chosen = rng.integers(0, len(unique), len(unique))
        means.append(sums[chosen].sum() / counts[chosen].sum())
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))
