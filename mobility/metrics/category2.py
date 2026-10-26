"""
Category 2: activity bout structure. See docs/category2.md.

All four metrics come from the active / inactive sequence of mobility.states.States, so radar and MoCap get the same
rules. The bouts follow the rule in States.extra['bout_rule'] (mobility.bouts.BoutRule, calibrated against the
activity labels without the evaluated person; mobility.evaluation.metric_table sets it).

The fragmentation index is the active -> inactive transition probability on the epochs *before* the bout rules. On the
bouts themselves it would be exactly 1 / mean bout duration, so it would add nothing.
"""
import numpy as np

from ..bouts import bouts, transition_probability
from . import metric


def _durations(s):
    rule = s.extra.get('bout_rule')
    if rule is None:
        raise ValueError("Category 2 needs States.extra['bout_rule'] (mobility.bouts.bout_rules)")
    b = bouts(s.active, s.valid, rule)
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
    """active -> inactive transition probability per epoch (ASTP), before the bout rules"""
    return transition_probability(s.active, s.valid)
