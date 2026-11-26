"""
Per-epoch states, from the radar or from MoCap, in the same form, so every metric is one rule applied to both.

Category 1 and 2 need: movement energy, activity level, active / inactive. Category 2 corrects the radar level for
the distance to the person (distance_corrected).
Category 3 needs the posture per epoch (`posture`, codes of mobility.posture) and the postural transitions: events, not
epochs (mobility.transitions). mobility.evaluation.metric_table sets both.
"""
from dataclasses import dataclass, field, replace

import warnings

import numpy as np
import pandas as pd

from .bouts import SPEED_FEATURE
from .config import EPOCH_FRAMES, EPOCH_S, MAX_MOCAP_RMS, RADAR_XYZ
from .groundtruth import active_speed_thresholds
from .radar.features import FEATURES
from .utils import epoch_mean, fold_of, log_level


@dataclass
class States:
    source: str              # 'radar' or 'mocap'
    t: np.ndarray            # (E,) start time of each epoch, epoch s
    energy: np.ndarray       # (E,) movement energy per epoch (linear; units depend on the source)
    level: np.ndarray        # (E,) activity level the active decision is made on
    active: np.ndarray       # (E,) bool
    valid: np.ndarray        # (E,) bool: epoch inside the evaluation window
    epoch_s: float = EPOCH_S
    posture: np.ndarray = None               # (E,) posture code per epoch (mobility.posture), Category 3
    transitions: object = None               # pd.DataFrame of postural transitions (mobility.transitions), Category 3
    extra: dict = field(default_factory=dict)   # anything else a metric needs (e.g. the bout rule)


def _epochs(session, window):
    n = EPOCH_FRAMES
    t = session.t[::n][:len(session.t) // n]
    return n, t, session.epoch_mask(window, n)


def feature_level(session, name, n=None):
    """per-epoch level of a radar feature: epoch mean of its log level (log features) or of the value"""
    n = n or EPOCH_FRAMES
    x = session.radar[name]
    return epoch_mean(log_level(x) if FEATURES[name].log else x, n)


def epoch_speed(session, n=None, name=SPEED_FEATURE):
    """(E,) per-epoch radar speed (m/s) for the Category 2 speed confirmation: the mean of `name` over the frames of
    the epoch that have a value (frames with moving points); 0 where no frame has one (nothing seen moving)"""
    n = n or EPOCH_FRAMES
    x = np.asarray(session.radar[name], float)
    T = len(x) // n * n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN epochs
        v = np.nanmean(x[:T].reshape(-1, n), axis=1)
    return np.nan_to_num(v, nan=0.0)


# ---- distance to the person (Category 2: the radar level corrected for it) ----

DISTANCE_REF_M = 2.5    # the corrected level equals the measured one at this distance (only shifts it: the thresholds
                        # are calibrated on the corrected level)


def radar_distance(session):
    """(T,) distance (m) from the radar to the middle of the person, per frame, from the radar's own estimate of their
    position (x_person, z_person) and the median height of their points (h_med, angle pass). Causal: a 1 s trailing
    median, carried forward while the person isn't seen; DISTANCE_REF_M before they are first seen."""
    n = EPOCH_FRAMES
    cols = []
    for k in ('x_person', 'z_person', 'h_med'):
        c = pd.Series(np.asarray(session.radar[k], float)).rolling(n, min_periods=n // 4).median().ffill()
        cols.append(c.values)
    x, z, h = cols
    d = np.sqrt((x - RADAR_XYZ[0]) ** 2 + (z - RADAR_XYZ[2]) ** 2 + (RADAR_XYZ[1] - h) ** 2)
    return np.where(np.isnan(d), DISTANCE_REF_M, d)


def log_distance(session, level_feature, n=None):
    """(E,) per epoch: log10(radar_distance / DISTANCE_REF_M), averaged over the frames where the level feature is
    defined, so that level + slope * log_distance is the epoch level of the corrected per-frame level"""
    n = n or EPOCH_FRAMES
    ld = np.log10(radar_distance(session) / DISTANCE_REF_M)
    ld[np.isnan(np.asarray(session.radar[level_feature], float))] = np.nan
    return epoch_mean(ld, n)


def distance_corrected(rs, slope, threshold):
    """radar States with the level corrected for the distance to the person, level + slope * extra['log_distance']
    (the level falls with distance: farther away fewer cells rise above the noise floor for the same movement), and
    active above `threshold`, an entry threshold calibrated on the corrected level. Epochs without a distance stay as
    they are."""
    level = rs.level + slope * np.nan_to_num(rs.extra['log_distance'])
    extra = dict(rs.extra, threshold=threshold, distance_slope=slope)
    return replace(rs, level=level, active=level > threshold, extra=extra)


def radar_states(session, threshold, active_feature='n_cells', energy_feature='e_move', window='window'):
    """active when the epoch level of active_feature exceeds threshold (calibrated against MoCap); energy from
    energy_feature (e_move: mean squared radial speed of the person's points plus their horizontal speed squared,
    from the angle pass). From the angle pass, when present: extra['speed'], epoch_speed (Category 2 option);
    extra['log_distance'] (log_distance) and extra['seen'], the share of the epoch's frames where the radar sees the
    person (Category 2's distance correction)."""
    n, t, valid = _epochs(session, window)
    level = feature_level(session, active_feature, n)
    energy = epoch_mean(session.radar[energy_feature], n)
    valid &= ~np.isnan(level) & ~np.isnan(energy)
    extra = dict(active_feature=active_feature, energy_feature=energy_feature, threshold=threshold)
    if SPEED_FEATURE in session.radar:
        extra['speed'] = epoch_speed(session, n)[:len(t)]
    if 'h_med' in session.radar:
        extra['log_distance'] = log_distance(session, active_feature, n)[:len(t)]
        extra['seen'] = epoch_mean(np.asarray(session.radar['n_person'], float) > 0, n)[:len(t)]
    return States('radar', t, energy, level, level > threshold, valid, extra=extra)


def mocap_states(session, active_speed=None, window='window'):
    """energy: mean squared joint speed; level: RMS joint speed (m/s); active above active_speed. Default
    active_speed: the threshold calibrated against the activity labels without this person
    (scripts/calibrate_mocap_threshold.py). Epochs faster than MAX_MOCAP_RMS are tracking jumps: not valid."""
    if active_speed is None:
        active_speed = fold_of(active_speed_thresholds(), session.person)
    n, t, valid = _epochs(session, window)
    energy = epoch_mean(session.mocap['ke'], n)
    level = np.sqrt(energy)
    valid &= ~np.isnan(level) & (level <= MAX_MOCAP_RMS)
    return States('mocap', t, energy, level, level > active_speed, valid, extra=dict(active_speed=active_speed))
