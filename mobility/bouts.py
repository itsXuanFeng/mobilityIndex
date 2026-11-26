"""
Activity bouts from a per-epoch activity level (Category 2). The same steps are applied to the radar and to MoCap:
    1. hysteresis on the level: an epoch above the entry threshold starts activity, which lasts while the level stays
       above a lower exit threshold;
    2. confirmation: a run of activity counts only if its level exceeds a higher confirmation threshold at least once
       (drops weak movements, such as arm movements, that just cross the entry threshold), or if it contains a posture
       transition of the same sensor (Category 3: a change of posture is a whole-body movement, even when the level
       stays low, as when sitting up under a blanket). The run keeps its start at the entry threshold. Optionally
       (radar only) a level-confirmed run must also reach a confirmation speed: its per-epoch Doppler speed
       (States.extra['speed'], SPEED_FEATURE) exceeds confirm_speed at least once;
    3. gaps: inactive runs of at most max_gap epochs between two active runs are filled;
    4. minimum bout: active runs shorter than min_bout epochs (after 3) are dropped.
Each step uses only past epochs; online, a bout is known once it is confirmed, and dated back to its start (a
transition is known once the posture model has decided it, see docs/category3.md).
Epochs outside the evaluation window (valid = False) end a bout: a bout never spans them and no gap is filled across
them. A bout cut by the window edge counts, with the part inside the window.

Nothing is set by hand. For MoCap, the entry is the Category 1 active threshold, and the exit and confirmation
thresholds (factors x the entry), max_gap and min_bout are calibrated together against the activity labels, leaving
the evaluated person out (scripts/calibrate_bout_rule.py): the combination that, applied to the MoCap joint speed with
the MoCap posture transitions, best separates active from inactive label epochs (balanced accuracy). The radar level
is corrected for the distance to the person (mobility.states.distance_corrected; its entry threshold calibrated on the
corrected level as the Category 1 threshold is), and uses one margin kappa on it, the same below and above the entry
(exit = entry - kappa, confirmation = entry + kappa), calibrated against the MoCap bout count, also leaving the person
out (mobility.evaluation.loso_radar_bout_levels); with the speed option, kappa and confirm_speed together.

Radar and MoCap bouts are compared one to one (match_bouts), by intersection over union.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import ACTIVE, INACTIVE
from .utils import read_calibration, runs

BOUT_RULE_FILE = 'bout_rule.csv'   # in the bout_rule run (scripts/calibrate_bout_rule.py)
GRID_GAP = range(0, 4)        # max_gap candidates, epochs
GRID_MIN_BOUT = range(1, 5)   # min_bout candidates, epochs
GRID_EXIT = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)   # MoCap exit threshold / entry threshold; 1 = no hysteresis
GRID_CONFIRM = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5)     # MoCap confirmation threshold / entry; 1 = no confirmation
# balanced accuracies this close to the best count as tied: well below the spread between leave-person-out folds
# (~0.005), so the labels can't tell such candidates apart; the least smoothing among them is taken
LABEL_TOLERANCE = 0.001
BOUT_MATCH_IOU = 0.5   # a radar bout and a MoCap bout match when they overlap by at least this intersection over union
# radar speed for the optional speed confirmation (mobility.radar.angle): the power-weighted 95th percentile of the
# |radial velocity| of the person's moving points (m/s); per epoch, the mean over the frames that have one, 0 without
SPEED_FEATURE = 'v_q950'


@dataclass(frozen=True)
class BoutRule:
    max_gap: int     # epochs: inactive gaps up to this long inside a bout are filled
    min_bout: int    # epochs: shorter bouts are dropped

    def __str__(self):
        return f'gaps <= {self.max_gap} filled, bouts >= {self.min_bout}'


def _level(level):
    return np.nan_to_num(np.asarray(level, float), nan=-np.inf)


def hysteresis(active, level, valid, exit_threshold, confirm_threshold=None, speed=None, confirm_speed=None,
               transition=None):
    """(E,) bool: active from an epoch that is `active` (above the entry threshold) for as long as the level stays
    above exit_threshold (at most the entry threshold); an invalid epoch ends it. With exit_threshold equal to the
    entry threshold, this is `active` itself. With confirm_threshold: only the confirmed runs (confirmed; with
    confirm_speed also on speed, with transition also by a posture transition)."""
    valid = np.asarray(valid, bool)
    start = np.asarray(active, bool) & valid
    above = valid & (_level(level) > exit_threshold)
    out = np.zeros(len(start), bool)
    for s, e in runs(above):
        h = np.flatnonzero(start[s:e])
        if len(h):
            out[s + h[0]:e] = True
    out |= start
    return out if confirm_threshold is None else confirmed(out, level, confirm_threshold, speed, confirm_speed,
                                                           transition)


def confirmed(active, level, confirm_threshold, speed=None, confirm_speed=None, transition=None):
    """(E,) bool: the runs of `active` whose level exceeds confirm_threshold at least once (with confirm_speed, not
    None: whose speed, per epoch in m/s, also exceeds confirm_speed at least once, not necessarily in the same epoch),
    or, with transition ((E,) bool: the epoch contains a posture transition, transition_epochs), that contain a
    posture transition"""
    out = np.asarray(active, bool).copy()
    lv = _level(level)
    sp = None if confirm_speed is None else np.nan_to_num(np.asarray(speed, float), nan=0.0)
    tr = None if transition is None else np.asarray(transition, bool)
    for s, e in runs(out):
        by_level = lv[s:e].max() > confirm_threshold and (sp is None or sp[s:e].max() > confirm_speed)
        if not by_level and not (tr is not None and tr[s:e].any()):
            out[s:e] = False
    return out


def transition_epochs(t, times, epoch_s):
    """(E,) bool: the epoch (start times t, epoch_s long) contains one of the event times (posture transitions)"""
    t = np.asarray(t, float)
    out = np.zeros(len(t), bool)
    times = np.asarray(times, float)
    i = np.searchsorted(t, times, side='right') - 1
    ok = (i >= 0) & (times < t[np.clip(i, 0, None)] + epoch_s)
    out[i[ok]] = True
    return out


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


def level_bouts(level, valid, entry, exit_threshold, confirm_threshold, rule, speed=None, confirm_speed=None,
                transition=None):
    """(n, 2) [start, end) epoch indices of the bouts of an activity level: active above the entry threshold, then
    hysteresis, confirmation (on the level, with confirm_speed also on speed; or by a posture transition) and the bout
    rule"""
    return bouts(hysteresis(level > entry, level, valid, exit_threshold, confirm_threshold, speed, confirm_speed,
                            transition), valid, rule)


def match_bouts(ref, est, min_iou=BOUT_MATCH_IOU):
    """one-to-one matching of (n, 2) and (m, 2) [start, end) bouts, greedily by intersection over union.
    Returns {index in ref: index in est}."""
    if not len(ref) or not len(est):
        return {}
    inter = np.clip(np.minimum(ref[:, None, 1], est[None, :, 1]) - np.maximum(ref[:, None, 0], est[None, :, 0]), 0,
                    None)
    union = (ref[:, None, 1] - ref[:, None, 0]) + (est[None, :, 1] - est[None, :, 0]) - inter
    iou = inter / union
    out, used = {}, set()
    for k in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(k, iou.shape)
        if iou[i, j] < min_iou:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def transition_probability(active, valid):
    """transition probability out of a state per epoch: the share of epochs in `active` that are followed by an epoch
    not in it, over pairs of consecutive valid epochs. NaN without such epochs. On an active sequence this is the
    active -> inactive probability (ASTP); on the epochs outside bouts (~in_bout), the non-bout -> bout probability."""
    a, v = np.asarray(active, bool), np.asarray(valid, bool)
    pair = v[:-1] & v[1:] & a[:-1]
    return float((~a[1:][pair]).mean()) if pair.any() else np.nan


# ---- calibration against the activity labels ----

def rule_confusion(epochs, thresholds, transitions=None, grid_gap=GRID_GAP, grid_min_bout=GRID_MIN_BOUT,
                   grid_exit=GRID_EXIT, grid_confirm=GRID_CONFIRM):
    """for every recording in epochs (mocap_epochs.csv: person, scenario, t, speed, cls), every candidate (exit
    factor, confirmation factor, max_gap, min_bout) and every fold threshold in thresholds ({left-out person: MoCap
    active threshold}): counts of label epochs (active / inactive class) inside and outside bouts. With transitions
    ({(person, scenario): {fold: (E,) bool, epochs of the recording with a MoCap posture transition under that fold's
    posture rule}}), a run with a transition is confirmed too. One row per (fold, person, scenario, exit_factor,
    confirm_factor, max_gap, min_bout)."""
    rows = []
    for (person, scenario), g in epochs.groupby(['person', 'scenario'], sort=False):
        g = g.sort_values('t')
        speed, cls = g.speed.values, g.cls.values
        valid = ~np.isnan(speed)
        y_act, y_ina = cls == ACTIVE, cls == INACTIVE
        done = {}                                # fold thresholds barely differ: most give the same sequence
        for fold, thr in thresholds.items():
            active = valid & (np.nan_to_num(speed) > thr)
            tr = None if transitions is None else transitions[(person, scenario)][fold]
            key = active.tobytes() + (b'' if tr is None else tr.tobytes())
            if key not in done:
                done[key] = []
                for f in grid_exit:
                    h = hysteresis(active, speed, valid, f * thr)
                    for c in grid_confirm:
                        hc = confirmed(h, speed, c * thr, transition=tr)
                        for gap in grid_gap:
                            for mb in grid_min_bout:
                                b = in_bouts(hc, valid, BoutRule(gap, mb))
                                done[key].append((f, c, gap, mb, (b & y_act).sum(), (~b & y_act).sum(),
                                                  (~b & y_ina).sum(), (b & y_ina).sum()))
            rows += [(fold, person, scenario) + r for r in done[key]]
    return pd.DataFrame(rows, columns=['fold', 'person', 'scenario', 'exit_factor', 'confirm_factor', 'max_gap',
                                       'min_bout', 'tp', 'fn', 'tn', 'fp'])


def balanced_accuracy(c):
    return (c.tp / (c.tp + c.fn) + c.tn / (c.tn + c.fp)) / 2


def calibrate_bout_rule(confusion, exclude_person=None, tolerance=LABEL_TOLERANCE):
    """(BoutRule, MoCap exit factor, MoCap confirmation factor, balanced accuracy): among the combinations within
    tolerance of the highest balanced accuracy on the label epochs of all persons except exclude_person (MoCap active
    defined by exclude_person's fold threshold; 'ALL' when None), the least smoothing: lowest confirmation, then least
    hysteresis, then smallest max_gap, then smallest min_bout."""
    fold = exclude_person or 'ALL'
    c = confusion[(confusion.fold == fold) & (confusion.person != exclude_person)]
    ba = balanced_accuracy(c.groupby(['exit_factor', 'confirm_factor', 'max_gap', 'min_bout'])[
        ['tp', 'fn', 'tn', 'fp']].sum())
    f, cf, gap, mb = sorted(ba[ba >= ba.max() - tolerance].index, key=lambda k: (k[1], -k[0], k[2], k[3]))[0]
    return BoutRule(int(gap), int(mb)), float(f), float(cf), float(ba[(f, cf, gap, mb)])


def bout_rules(path=None, epoch_s=1.0):
    """{person: BoutRule calibrated without that person, 'ALL': on every person}, from the latest run of
    scripts/calibrate_bout_rule.py (or `path`)"""
    d = read_calibration('bout_rule', BOUT_RULE_FILE, path)
    return {r.person: BoutRule(int(round(r.max_gap_s / epoch_s)), int(round(r.min_bout_s / epoch_s)))
            for r in d.itertuples()}


def mocap_level_factors(path=None):
    """{person: (exit, confirmation) threshold of MoCap / its entry threshold, calibrated without that person; 'ALL':
    on every person}"""
    d = read_calibration('bout_rule', BOUT_RULE_FILE, path)
    return {r.person: (r.mocap_exit_factor, r.mocap_confirm_factor) for r in d.itertuples()}
