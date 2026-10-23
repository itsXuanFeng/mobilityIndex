"""
Per-epoch states, from the radar or from MoCap, in the same form, so every metric is one rule applied to both.

Category 1 and 2 need: movement energy, activity level, active / inactive.
Category 3 will add posture (lying / sitting / standing) and transition events: add fields here and fill them in both
builders.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import EPOCH_S, FPS
from .groundtruth import active_speed_thresholds
from .radar.features import FEATURES
from .utils import epoch_mean, log_level


@dataclass
class States:
    source: str              # 'radar' or 'mocap'
    t: np.ndarray            # (E,) start time of each epoch, epoch s
    energy: np.ndarray       # (E,) movement energy per epoch (linear; units depend on the source)
    level: np.ndarray        # (E,) activity level the active decision is made on
    active: np.ndarray       # (E,) bool
    valid: np.ndarray        # (E,) bool: epoch inside the evaluation window
    epoch_s: float = EPOCH_S
    extra: dict = field(default_factory=dict)   # anything else a metric needs (e.g. labels, posture)


def _epochs(session, window):
    n = int(round(EPOCH_S * FPS))
    t = session.t[::n][:len(session.t) // n]
    valid = epoch_mean(session.masks[window], n) >= 0.5
    return n, t, valid


def feature_level(session, name, n=None):
    """per-epoch level of a radar feature: epoch mean of its log level (log features) or of the value"""
    n = n or int(round(EPOCH_S * FPS))
    x = session.radar[name]
    return epoch_mean(log_level(x) if FEATURES[name].log else x, n)


def radar_states(session, threshold, active_feature='n_cells', energy_feature='p_r4', window='window'):
    """active when the epoch level of active_feature exceeds threshold (calibrated against MoCap); energy from
    energy_feature (p_r4: distance-corrected power)"""
    n, t, valid = _epochs(session, window)
    level = feature_level(session, active_feature, n)
    energy = epoch_mean(session.radar[energy_feature], n)
    valid &= ~np.isnan(level) & ~np.isnan(energy)
    return States('radar', t, energy, level, level > threshold, valid,
                  extra=dict(active_feature=active_feature, energy_feature=energy_feature, threshold=threshold))


def mocap_states(session, active_speed=None, window='window'):
    """energy: mean squared joint speed; level: RMS joint speed (m/s); active above active_speed. Default
    active_speed: the threshold calibrated against the activity labels without this person
    (scripts/calibrate_mocap_threshold.py)."""
    if active_speed is None:
        thr = active_speed_thresholds()
        active_speed = thr.get(session.person, thr['ALL'])   # 'ALL': people outside the calibration
    n, t, valid = _epochs(session, window)
    energy = epoch_mean(session.mocap['ke'], n)
    level = np.sqrt(energy)
    valid &= ~np.isnan(level)
    return States('mocap', t, energy, level, level > active_speed, valid, extra=dict(active_speed=active_speed))
