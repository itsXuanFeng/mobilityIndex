"""Small NaN-aware signal helpers."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def log_level(x):
    """log10 of a non-negative signal, offset by its 5th percentile (or its smallest positive value) so that frames
    near zero don't dominate: a recording-level, monotonic transform"""
    x = np.asarray(x, float)
    off = np.nanpercentile(x, 5)
    if not off > 0:
        pos = x[x > 0]
        off = pos.min() if pos.size else 1.0
    return np.log10(x + off)


def nan_smooth(x, n):
    """moving average over n samples that skips NaN (NaN where less than half the window is valid)"""
    m = np.isnan(x)
    a = uniform_filter1d(np.where(m, 0, x), n)
    b = uniform_filter1d((~m).astype(float), n)
    return np.where(b > 0.5, a / np.maximum(b, 1e-9), np.nan)


def best_threshold(x, y):
    """threshold on x that maximises the balanced accuracy for the boolean labels y (NaN in x ignored)"""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    cands = np.unique(np.percentile(x, np.linspace(1, 99, 197)))
    ba = [((x[y] > c).mean() + (x[~y] <= c).mean()) / 2 for c in cands]
    return float(cands[int(np.argmax(ba))])


def epoch_mean(x, n):
    """mean over consecutive epochs of n samples; NaN where less than half the samples are valid. Booleans give the
    fraction of True."""
    x = np.asarray(x, float)
    T = len(x) // n * n
    x = x[:T].reshape(-1, n)
    ok = (~np.isnan(x)).sum(1) >= n / 2
    s = np.where(np.isnan(x), 0, x).sum(1)
    c = np.maximum((~np.isnan(x)).sum(1), 1)
    return np.where(ok, s / c, np.nan)
