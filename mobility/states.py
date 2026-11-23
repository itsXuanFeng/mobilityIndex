"""
Per-epoch states, from the radar or from MoCap, in the same form, so every metric is one rule applied to both.

Category 1 and 2 need: movement energy, activity level, active / inactive.
Category 3 needs the posture per epoch (`posture`, codes of mobility.posture) and the postural transitions: events, not
epochs (mobility.transitions). mobility.evaluation.metric_table sets both.
"""
from dataclasses import dataclass, field

import warnings

import numpy as np

from .bouts import SPEED_FEATURE
from .config import EPOCH_FRAMES, EPOCH_S, MAX_MOCAP_RMS
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


def radar_states(session, threshold, active_feature='n_cells', energy_feature='e_move', window='window'):
    """active when the epoch level of active_feature exceeds threshold (calibrated against MoCap); energy from
    energy_feature (e_move: mean squared radial speed of the person's points plus their horizontal speed squared,
    from the angle pass). extra['speed']: epoch_speed, when the angle pass provides it (Category 2 option)."""
    n, t, valid = _epochs(session, window)
    level = feature_level(session, active_feature, n)
    energy = epoch_mean(session.radar[energy_feature], n)
    valid &= ~np.isnan(level) & ~np.isnan(energy)
    extra = dict(active_feature=active_feature, energy_feature=energy_feature, threshold=threshold)
    if SPEED_FEATURE in session.radar:
        extra['speed'] = epoch_speed(session, n)[:len(t)]
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
