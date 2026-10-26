"""
Activity bouts from a per-epoch active / inactive sequence (Category 2). The same rule is applied to the radar and to
MoCap.

A bout is a run of active epochs after two rules, a temporal hysteresis (a change of state has to last to count):
    1. gaps: inactive runs of at most max_gap epochs between two active runs are filled;
    2. minimum bout: active runs shorter than min_bout epochs (after 1) are dropped.
Epochs outside the evaluation window (valid = False) end a bout: a bout never spans them and no gap is filled across
them. A bout cut by the window edge counts, with the part inside the window.

The rule is not set by hand. Like the MoCap active threshold, it is calibrated against the activity labels, leaving
the evaluated person out (scripts/calibrate_bout_rule.py): the rule that, applied to the MoCap active sequence, best
separates active from inactive label epochs (balanced accuracy).
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GROUNDTRUTH_DIR
from .labels import ACTIVE, INACTIVE

BOUT_RULE_FILE = os.path.join(GROUNDTRUTH_DIR, 'bout_rule.csv')
GRID_GAP = range(0, 11)       # max_gap candidates, epochs
GRID_MIN_BOUT = range(1, 11)  # min_bout candidates, epochs


@dataclass(frozen=True)
class BoutRule:
    max_gap: int     # epochs: inactive gaps up to this long inside a bout are filled
    min_bout: int    # epochs: shorter bouts are dropped

    def __str__(self):
        return f'gaps <= {self.max_gap} filled, bouts >= {self.min_bout}'


def runs(mask):
    """(n, 2) int array of [start, end) of the runs of True"""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return d.reshape(-1, 2)


def in_bouts(active, valid, rule):
    """(E,) bool: epoch inside a bout"""
    valid = np.asarray(valid, bool)
    a = np.asarray(active, bool) & valid
    out = a.copy()
    if rule.max_gap > 0:
        for s, e in runs(valid & ~a):            # inactive runs; an invalid epoch ends the run
            if e - s <= rule.max_gap and s > 0 and e < len(a) and a[s - 1] and a[e]:
                out[s:e] = True
    if rule.min_bout > 1:
        for s, e in runs(out):
            if e - s < rule.min_bout:
                out[s:e] = False
    return out


def bouts(active, valid, rule):
    """(n, 2) [start, end) epoch indices of the bouts"""
    return runs(in_bouts(active, valid, rule))


def transition_probability(active, valid):
    """active -> inactive transition probability per epoch (ASTP), on the sequence before the bout rules: the share of
    active epochs followed by an inactive one, over pairs of consecutive valid epochs. NaN without active epochs."""
    a, v = np.asarray(active, bool), np.asarray(valid, bool)
    pair = v[:-1] & v[1:] & a[:-1]
    return float((~a[1:][pair]).mean()) if pair.any() else np.nan


# ---- calibration against the activity labels ----

def rule_confusion(epochs, thresholds, grid_gap=GRID_GAP, grid_min_bout=GRID_MIN_BOUT):
    """for every recording in epochs (mocap_epochs.csv: person, scenario, t, speed, cls), every candidate rule and
    every fold threshold in thresholds ({left-out person: MoCap active threshold}): counts of label epochs (active /
    inactive class) inside and outside bouts. One row per (fold, person, scenario, max_gap, min_bout)."""
    rows = []
    for (person, scenario), g in epochs.groupby(['person', 'scenario'], sort=False):
        g = g.sort_values('t')
        speed, cls = g.speed.values, g.cls.values
        valid = ~np.isnan(speed)
        y_act, y_ina = cls == ACTIVE, cls == INACTIVE
        done = {}                                # fold thresholds barely differ: most give the same sequence
        for fold, thr in thresholds.items():
            active = valid & (np.nan_to_num(speed) > thr)
            key = active.tobytes()
            if key not in done:
                done[key] = []
                for gap in grid_gap:
                    for mb in grid_min_bout:
                        b = in_bouts(active, valid, BoutRule(gap, mb))
                        done[key].append((gap, mb, (b & y_act).sum(), (~b & y_act).sum(), (~b & y_ina).sum(),
                                          (b & y_ina).sum()))
            rows += [(fold, person, scenario) + r for r in done[key]]
    return pd.DataFrame(rows, columns=['fold', 'person', 'scenario', 'max_gap', 'min_bout', 'tp', 'fn', 'tn', 'fp'])


def balanced_accuracy(c):
    return (c.tp / (c.tp + c.fn) + c.tn / (c.tn + c.fp)) / 2


def calibrate_bout_rule(confusion, exclude_person=None):
    """the rule with the highest balanced accuracy on the label epochs of all persons except exclude_person, with
    MoCap active defined by exclude_person's fold threshold ('ALL' when None). Ties: the simplest rule (smallest
    max_gap, then smallest min_bout)."""
    fold = exclude_person or 'ALL'
    c = confusion[(confusion.fold == fold) & (confusion.person != exclude_person)]
    ba = balanced_accuracy(c.groupby(['max_gap', 'min_bout'])[['tp', 'fn', 'tn', 'fp']].sum())
    gap, mb = ba[ba == ba.max()].index.min()
    return BoutRule(int(gap), int(mb)), float(ba.max())


def bout_rules(path=BOUT_RULE_FILE, epoch_s=1.0):
    """{person: BoutRule calibrated without that person, 'ALL': on every person}, from scripts/calibrate_bout_rule.py"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run scripts/calibrate_bout_rule.py first')
    d = pd.read_csv(path)
    return {r.person: BoutRule(int(round(r.max_gap_s / epoch_s)), int(round(r.min_bout_s / epoch_s)))
            for r in d.itertuples()}
