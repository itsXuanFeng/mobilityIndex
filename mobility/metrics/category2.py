"""
Category 2: activity bout structure. See docs/category2.md.

All four metrics come from the activity level of mobility.states.States, so radar and MoCap get the same steps
(mobility.bouts): hysteresis on the level, from the Category 1 active threshold down to the exit threshold in
States.extra['exit_threshold']; confirmation, keeping the runs that exceed States.extra['confirm_threshold']; then the
bout rule in States.extra['bout_rule'] (gap filling, minimum bout). All are calibrated without the evaluated person;
mobility.evaluation.metric_table sets them.

The fragmentation index is the active -> inactive transition probability after the hysteresis and the confirmation,
*before* the bout rule.
On the bouts themselves it would be exactly 1 / mean bout duration, so it would add nothing.
"""
import numpy as np

from ..bouts import hysteresis, in_bouts, runs, transition_probability
from . import metric


def _need(s, key):
    if key not in s.extra:
        raise ValueError(f"Category 2 needs States.extra['{key}'] (mobility.evaluation.metric_table)")
    return s.extra[key]


def bout_activity(s):
    """(E,) bool: active after the hysteresis and the confirmation, before the bout rule"""
    return hysteresis(s.active, s.level, s.valid, _need(s, 'exit_threshold'), _need(s, 'confirm_threshold'))


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
    """active -> inactive transition probability per epoch (ASTP), after the hysteresis and the confirmation, before
    the bout rule"""
    return transition_probability(bout_activity(s), s.valid)
