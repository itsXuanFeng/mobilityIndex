"""
Category 2: activity bout structure. See docs/category2.md.

All four metrics come from the activity level of mobility.states.States, so radar and MoCap get the same steps
(mobility.bouts): hysteresis on the level, from the Category 1 active threshold down to the exit threshold in
States.extra['exit_threshold']; confirmation, keeping the runs that exceed States.extra['confirm_threshold'] (and, if
States.extra['confirm_speed'] is set, radar only, whose speed States.extra['speed'] exceeds it); then the bout rule in
States.extra['bout_rule'] (gap filling, minimum bout). All are calibrated without the evaluated person;
mobility.evaluation.metric_table sets them (set_bout_levels).

The fragmentation index is the non-bout -> bout transition probability: how often rest is interrupted by a new bout,
about 1 / mean rest between bouts. The other direction, bout -> non-bout, would be about 1 / mean bout duration and
would repeat bout_duration_mean.
"""
import numpy as np

from ..bouts import hysteresis, in_bouts, transition_probability
from ..utils import runs
from . import metric


def set_bout_levels(rs, ms, radar, mocap, radar_confirm_speed=None):
    """the Category 2 thresholds of each sensor (radar and MoCap States): radar and mocap are (exit, confirmation)
    thresholds on their level; radar_confirm_speed (m/s, None = no speed confirmation) on the radar speed. Always set,
    so a value from an earlier call never carries over."""
    rs.extra['exit_threshold'], rs.extra['confirm_threshold'] = radar
    ms.extra['exit_threshold'], ms.extra['confirm_threshold'] = mocap
    rs.extra['confirm_speed'], ms.extra['confirm_speed'] = radar_confirm_speed, None


def _need(s, key):
    if key not in s.extra:
        raise ValueError(f"Category 2 needs States.extra['{key}'] (mobility.evaluation.metric_table)")
    return s.extra[key]


def bout_activity(s):
    """(E,) bool: active after the hysteresis and the confirmation, before the bout rule"""
    c = s.extra.get('confirm_speed')
    if c is not None and s.extra.get('speed') is None:
        raise ValueError("a confirmation speed needs States.extra['speed'] (mobility.states.radar_states)")
    return hysteresis(s.active, s.level, s.valid, _need(s, 'exit_threshold'), _need(s, 'confirm_threshold'),
                      s.extra.get('speed'), c)


def in_bout(s):
    """(E,) bool: epoch inside a bout"""
    return in_bouts(bout_activity(s), s.valid, _need(s, 'bout_rule'))


def _durations(s):
    b = runs(in_bout(s))
    return (b[:, 1] - b[:, 0]) * s.epoch_s


@metric('n_bouts', category=2, unit='count')
def n_bouts(s):
    """number of active bouts in the valid epochs"""
    return float(len(_durations(s)))


@metric('bout_duration_mean', category=2, unit='s')
def bout_duration_mean(s):
    """mean bout duration; NaN without bouts"""
    d = _durations(s)
    return float(d.mean()) if len(d) else np.nan


@metric('bout_duration_var', category=2, unit='s^2')
def bout_duration_var(s):
    """variance of the bout duration (sample variance); NaN with fewer than 2 bouts"""
    d = _durations(s)
    return float(d.var(ddof=1)) if len(d) > 1 else np.nan


@metric('fragmentation_index', category=2, unit='per epoch')
def fragmentation_index(s):
    """non-bout -> bout transition probability per epoch: the share of valid epochs outside a bout that are followed by
    an epoch inside one (rest interrupted by a new bout)"""
    return transition_probability(~in_bout(s), s.valid)
