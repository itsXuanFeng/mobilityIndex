"""Small helpers: NaN-aware signals, runs of a mask, thresholds, calibration tables, summaries."""
import os

import numpy as np
import pandas as pd
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


def runs(mask):
    """(n, 2) int array of [start, end) of the runs of True"""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return d.reshape(-1, 2)


def best_threshold(x, y):
    """threshold on x that maximises the balanced accuracy for the boolean labels y (NaN in x ignored)"""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    cands = np.unique(np.percentile(x, np.linspace(1, 99, 197)))
    ba = [((x[y] > c).mean() + (x[~y] <= c).mean()) / 2 for c in cands]
    return float(cands[int(np.argmax(ba))])


# ---- leave-one-subject-out calibration tables ----

def read_calibration(path, script):
    """a calibration table (csv) written by `script`; FileNotFoundError telling to run it when missing"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run {script} first')
    return pd.read_csv(path)


def fold_of(per_person, person):
    """the value calibrated without `person`; 'ALL' (calibrated on every person) for people outside the calibration"""
    return per_person.get(person, per_person['ALL'])


def person_of(rec):
    """person of each recording name 'person_x/scenario_y' (pandas Series of names)"""
    return rec.str.split('/').str[0]


def spread(x, fmt='.3f'):
    """median [10th-90th percentile], for printing"""
    q = np.nanpercentile(x, [50, 10, 90])
    return f'{q[0]:{fmt}} [{q[1]:{fmt}}-{q[2]:{fmt}}]'
