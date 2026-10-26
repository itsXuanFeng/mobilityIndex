"""
Category 3: postural transitions, and the time in each posture. See docs/category3.md.

Posture times read the posture per epoch in States.posture (MoCap: the calibrated MoCap rule; radar: the height rule,
mobility.posture.RadarPostureRule), over the valid epochs where the posture is known.

The metrics read the transition events in States.transitions (mobility.transitions: MoCap from the calibrated posture
rule, radar from the height rule; mobility.evaluation.metric_table sets them) and count only those inside the
valid epochs, which are the same on both sides.

Peak transition Doppler is the mean, over the recording's counted transitions (sit-to-stand, stand-to-sit,
lie-to-sit), of the peak radial speed around each one: radar, a power-weighted quantile of the person's moving
speeds (calibrated against MoCap); MoCap, the fastest joint along the line to the radar. Both are speeds along the line
to the radar, so they compare (SCOPE.md).
"""
import numpy as np

from ..posture import POSTURES
from ..transitions import COUNTED
from . import metric


def _events(s, kinds=None):
    ev = s.transitions
    if ev is None:
        raise ValueError('Category 3 needs States.transitions (mobility.evaluation.metric_table(..., transitions=))')
    i = np.clip(((ev.t.values - s.t[0]) // s.epoch_s).astype(int), 0, len(s.valid) - 1)
    ev = ev[s.valid[i] & (ev.t.values >= s.t[0])]
    return ev if kinds is None else ev[ev.kind.isin(kinds)]


@metric('sit_to_stand_count', category=3, unit='count')
def sit_to_stand_count(s):
    """number of sit-to-stand transitions"""
    return float(len(_events(s, ['sit_to_stand'])))


@metric('stand_to_sit_count', category=3, unit='count')
def stand_to_sit_count(s):
    """number of stand-to-sit transitions"""
    return float(len(_events(s, ['stand_to_sit'])))


@metric('lie_to_sit_count', category=3, unit='count')
def lie_to_sit_count(s):
    """number of lie-to-sit transitions (in bed: sitting up, getting up to the edge)"""
    return float(len(_events(s, ['lie_to_sit'])))


@metric('peak_transition_doppler', category=3, unit='m/s')
def peak_transition_doppler(s):
    """mean peak radial speed of the counted transitions; NaN without any"""
    v = _events(s, COUNTED).peak_v.values
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


def _posture_time(s, code):
    if s.posture is None:
        raise ValueError('posture times need States.posture (mobility.evaluation.metric_table(..., postures=))')
    return float((s.valid & (s.posture == code)).sum() * s.epoch_s / 60)


def _register_posture_time(code, name):
    def posture_time(s):
        return _posture_time(s, code)
    posture_time.__doc__ = f'time {name} (valid epochs whose posture is {name})'
    metric(f'{name}_time', category=3, unit='min')(posture_time)


for _code, _name in enumerate(POSTURES):
    _register_posture_time(_code, _name)
