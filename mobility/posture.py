"""
Posture (Category 3): standing, sitting, lying, or on the floor, per sample.

From the script. Every scripted postural change (LABEL_TRANSITION) fixes the posture until the next one, so the labels
give a posture timeline: unknown inside a transition label and within LABEL_MARGIN_S of it, standing before the first
one. The floor tasks give FLOOR: lying, sitting or kneeling on the ground.

From the MoCap skeleton, by a rule calibrated against the script, leaving the evaluated person out
(scripts/calibrate_posture.py); nothing is set by hand:
    standing    pelvis above hip_stand
    floor       otherwise, pelvis or knees below low_floor        (lying, sitting or kneeling on the ground)
    lying       otherwise, trunk more than trunk_lie from vertical
    sitting     otherwise                                          (chair, edge of the bed, upright in bed)
From the radar, with the angle: the height of the top of the body (RadarPostureRule, below), thresholds calibrated
against the MoCap posture, leaving the person out. A posture counts only once it has lasted min_hold_s, for MoCap and
radar alike; mobility.transitions turns the changes into events.

hip_stand and trunk_lie maximise the balanced accuracy per 1 s epoch against the scripted posture. low_floor and
min_hold_s maximise how well the resulting transitions reproduce the scripted changes (F1). The floor is calibrated on
the transitions because per epoch it is decided by label noise: the few floor epochs where the person is already
getting up push it to the top of the wide gap between the floor (< 0.2 m) and sitting on furniture (> 0.43 m), where it
turns sitting on a low bed edge into "floor".
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EPOCH_S, GROUNDTRUTH_DIR, LABEL_MARGIN_S, OUTPUT_ROOT
from .groundtruth import mocap_50hz
from .io.mocap import MOCAP_FPS
from .io.recordings import labels as load_labels
from .labels import caregiver_period, script_period
from .utils import best_threshold

STANDING, SITTING, LYING, FLOOR = 0, 1, 2, 3
UNKNOWN = -1
POSTURES = ('standing', 'sitting', 'lying', 'floor')
LEVEL = {STANDING: 3, SITTING: 2, LYING: 1, FLOOR: 0}     # height order: a transition to a lower level is "down"

POSTURE_RULE_FILE = os.path.join(GROUNDTRUTH_DIR, 'posture_rule.csv')
GRID_LOW_FLOOR = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)   # m
GRID_MIN_HOLD_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

# scripted postural changes: label -> (posture before, posture after). Picking up a coin bends the trunk but ends
# where it started: no change, but the posture is unknown during it.
LABEL_TRANSITION = {
    'sit down on the chair': (STANDING, SITTING),
    'stand up from the chair': (SITTING, STANDING),
    'sit down on the edge of the bed': (STANDING, SITTING),
    'lay down on the bed': (SITTING, LYING),
    'sit upright in the bed': (LYING, SITTING),
    'go from upright to lay on the bed': (SITTING, LYING),
    'get up and sit on the edge of the bed': (LYING, SITTING),
    'stand up from the bed': (SITTING, STANDING),
    'lay down on the ground': (STANDING, FLOOR),
    'get up and stand upright': (FLOOR, STANDING),
    'act like you pick up a coin from the floor': (STANDING, STANDING),
}


def label_transition(label):
    """(before, after) for a scripted postural change, else None"""
    return LABEL_TRANSITION.get(label.strip().rstrip('.').lower())


def label_transitions(labels):
    """the scripted postural changes of a recording: label, start, end, before, after (the coin task excluded)"""
    rows = []
    for r in labels.itertuples():
        tr = label_transition(r.label)
        if tr and tr[0] != tr[1]:
            rows.append(dict(label=r.label, start=r.start, end=r.end, frm=tr[0], to=tr[1]))
    return pd.DataFrame(rows, columns=['label', 'start', 'end', 'frm', 'to'])


def label_postures(t, labels, margin=LABEL_MARGIN_S):
    """(len(t),) posture code from the script; UNKNOWN inside or within `margin` of a postural label, outside the
    scripted period and during the caregiver visit"""
    t = np.asarray(t, float)
    out = np.full(len(t), STANDING)
    for r in labels.sort_values('start').itertuples():
        tr = label_transition(r.label)
        if tr:
            out[t >= r.start] = tr[1]
    for r in labels.itertuples():
        if label_transition(r.label):
            out[(t >= r.start - margin) & (t <= r.end + margin)] = UNKNOWN
    s0, s1 = script_period(labels)
    out[(t < s0) | (t > s1)] = UNKNOWN
    care = caregiver_period(labels)
    if care:
        out[(t >= care[0]) & (t <= care[1])] = UNKNOWN
    return out


@dataclass(frozen=True)
class PostureRule:
    hip_stand: float       # m: pelvis above this = standing
    low_floor: float       # m: pelvis or knees below this = on the floor
    trunk_lie: float       # degrees from vertical: above this = lying
    min_hold_s: float      # a posture has to last this long to count

    def classify(self, sig):
        """posture code per sample of MoCap signals (dict with hip_y, knee_y, trunk_deg); UNKNOWN where missing"""
        hip, trunk = np.asarray(sig['hip_y']), np.asarray(sig['trunk_deg'])
        low = np.fmin(hip, np.asarray(sig['knee_y']))
        out = np.where(hip > self.hip_stand, STANDING,
                       np.where(low < self.low_floor, FLOOR, np.where(trunk > self.trunk_lie, LYING, SITTING)))
        out[np.isnan(hip) | np.isnan(trunk) | np.isnan(low)] = UNKNOWN
        return out

    def __str__(self):
        return (f'standing: pelvis > {self.hip_stand:.2f} m; floor: pelvis or knees < {self.low_floor:.2f} m; '
                f'lying: trunk > {self.trunk_lie:.0f} deg; held >= {self.min_hold_s:g} s')


# ---- calibration against the script ----

def posture_epochs(person, scenario):
    """1 s epochs on the MoCap clock: start time, pelvis, knee and trunk signals (epoch means) and the scripted
    posture at the middle of the epoch"""
    sig = mocap_50hz(person, scenario)
    if sig is None:
        return None
    n = int(round(EPOCH_S * MOCAP_FPS))
    E = len(sig['t']) // n
    t = sig['t'][:E * n].reshape(E, n)[:, 0]
    mean = {k: sig[k][:E * n].reshape(E, n).mean(1) for k in ('hip_y', 'knee_y', 'trunk_deg')}
    post = label_postures(t + EPOCH_S / 2, load_labels(person, scenario))
    return pd.DataFrame(dict(person=person, scenario=scenario, t=t, **mean, posture=post))


def calibrate_thresholds(epochs, exclude_person=None):
    """pelvis threshold (standing vs the rest) and trunk threshold (lying vs sitting), each maximising the balanced
    accuracy of its split on the scripted epochs of all persons except exclude_person"""
    e = epochs[(epochs.posture != UNKNOWN) & (epochs.person != exclude_person)]
    hip_stand = best_threshold(e.hip_y.values, (e.posture == STANDING).values)
    furniture = e[e.posture.isin([SITTING, LYING])]
    trunk_lie = best_threshold(furniture.trunk_deg.values, (furniture.posture == LYING).values)
    return hip_stand, trunk_lie


def posture_rules(path=POSTURE_RULE_FILE):
    """{person: PostureRule calibrated without that person, 'ALL': on every person}, from scripts/calibrate_posture.py"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run scripts/calibrate_posture.py first')
    d = pd.read_csv(path)
    return {r.person: PostureRule(r.hip_stand, r.low_floor, r.trunk_lie, r.min_hold_s) for r in d.itertuples()}


# ---- from the radar: the height of the top of the body (angle pass) ----

HEIGHT_SMOOTH_S = 1.0                 # running median of the radar height over this long
GRID_OFFAXIS_K = (0.0, 0.05, 0.1, 0.15, 0.2)   # m of height per m from the radar axis (RadarPostureRule.offaxis_k)
RADAR_POSTURE_RULE_FILE = os.path.join(OUTPUT_ROOT, 'category3', 'radar_posture_rule.csv')   # evaluate_category3


@dataclass(frozen=True)
class RadarPostureRule:
    """posture from the radar's top-of-body height h (mobility.radar.angle.frame_features), the order of the levels:
        standing  h > h_stand;   sitting  h_sit < h <= h_stand;   lying  h_floor < h <= h_sit;   floor  h <= h_floor
    h is first corrected for the distance d of the person from the radar axis: h + offaxis_k * d. Off-axis the radar
    sees less of the top of the head (more of the chest, which breathes), so the measured height falls by ~0.1 m per m
    from the axis; the directions themselves are unbiased (docs/category3.md).
    Where the radar doesn't see the person, the last posture carries forward (a still, unseen person hasn't moved).
    A posture counts once it has lasted min_hold_s, as for MoCap. Thresholds: balanced accuracy per epoch against the
    MoCap posture; offaxis_k and min_hold_s: transition F1 against MoCap; all leaving the evaluated person out
    (evaluate_category3.py)."""
    h_stand: float
    h_sit: float
    h_floor: float
    min_hold_s: float
    offaxis_k: float = 0.0

    def height(self, height, offaxis):
        """the corrected height"""
        return np.asarray(height, float) + self.offaxis_k * np.nan_to_num(np.asarray(offaxis, float))

    def classify(self, height, offaxis):
        """posture code per frame from the radar height and distance from the axis (radar_height, radar_offaxis);
        UNKNOWN only before the person is first seen"""
        h = self.height(height, offaxis)
        out = np.where(h > self.h_stand, STANDING, np.where(h > self.h_sit, SITTING,
                                                            np.where(h > self.h_floor, LYING, FLOOR)))
        out = pd.Series(np.where(np.isnan(h), np.nan, out)).ffill().values
        return np.where(np.isnan(out), UNKNOWN, out).astype(int)

    def __str__(self):
        return (f'height + {self.offaxis_k:g} x distance from the axis: standing > {self.h_stand:.2f} m, sitting > '
                f'{self.h_sit:.2f} m, lying > {self.h_floor:.2f} m; held >= {self.min_hold_s:g} s')


def _running_median(x, fps=None):
    from .config import FPS
    n = int(round(HEIGHT_SMOOTH_S * (fps or FPS)))
    return pd.Series(x).rolling(n, center=True, min_periods=n // 2).median().values


def radar_height(session):
    """the radar's top-of-body height per frame, running median over HEIGHT_SMOOTH_S (NaN where the person isn't seen
    for most of that window)"""
    return _running_median(session.radar['h_top'])


def radar_offaxis(session):
    """the radar's horizontal distance of the person from the point below the radar, per frame, smoothed as the
    height"""
    from .config import RADAR_XYZ
    return _running_median(np.hypot(session.radar['x_person'] - RADAR_XYZ[0],
                                    session.radar['z_person'] - RADAR_XYZ[2]))


def epoch_posture(posture, n):
    """posture per epoch of n frames: the most frequent known posture, UNKNOWN when none is known"""
    p = np.asarray(posture)[:len(posture) // n * n].reshape(-1, n)
    counts = np.stack([(p == k).sum(1) for k in range(len(POSTURES))], 1)
    return np.where(counts.sum(1) > 0, counts.argmax(1), UNKNOWN)


def calibrate_radar_thresholds(epochs, exclude_person=None):
    """h_stand, h_sit, h_floor: each splits two neighbouring postures (standing | sitting | lying | floor) with the
    highest balanced accuracy, on the epochs (columns person, height, posture = MoCap) of all persons except
    exclude_person where the radar sees the person"""
    e = epochs[(epochs.person != exclude_person) & ~epochs.height.isna()]

    def split(upper, lower):
        g = e[e.posture.isin([upper, lower])]
        return best_threshold(g.height.values, (g.posture == upper).values)
    return split(STANDING, SITTING), split(SITTING, LYING), split(LYING, FLOOR)


def radar_posture_rules(path=RADAR_POSTURE_RULE_FILE):
    """{person: RadarPostureRule calibrated without that person, 'ALL': on every person}"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run scripts/evaluate_category3.py first')
    d = pd.read_csv(path)
    return {r.person: RadarPostureRule(r.h_stand, r.h_sit, r.h_floor, r.min_hold_s, r.offaxis_k)
            for r in d.itertuples()}
