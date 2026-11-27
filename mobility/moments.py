"""
Moments whose meaning is certain: stretches of unlabelled radar data that can only mean one thing, so they give
calibration information without labels or MoCap (the detailed notes, notes/docs/calibration.md, "Unsupervised calibration").

Walking is the one such moment found on the January data: the radar's track of the person moves faster than
WALK_SPEED over the whole trailing WALK_WINDOW_S, with at least WALK_MIN_PATH of net displacement over that window.
A person who walks stands and is active, by the definitions of Categories 1 and 3. Causal: each frame is judged on
the past WALK_WINDOW_S only. The settings are checked against MoCap by scripts/check_mounting_height.py (the share of
walking frames where MoCap says standing, and of walking epochs MoCap calls active). Staying in one place was tested
too and is not certain: MoCap calls part of it active and its posture is mixed.

What walking gives:
    walking height     the person's top-of-body and median height while walking, relative to the population's standing
                       height at the same distance from the radar axis (the posture model's standing emission mean):
                       stature, and an error in the radar's installed height, which shifts every measured height
    distance curve     the walking top-of-body height still depends on the horizontal distance from the radar axis, in
                       a way the posture model's straight line doesn't follow (within 2 cm up to 2.5 m, then -17 cm
                       at 2.5-3 m and +10 cm beyond 3 m on the January data, part of it the room's): DistanceCurve,
                       the population's median per 0.25 m of distance, is subtracted, so the walking height doesn't
                       depend on where in the room the person walks
    mounting check     the walking-height offset of a resident, compared with the spread of the population
                       (MountingCheck, calibrated leaving the person out by scripts/check_mounting_height.py)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FPS
from .posture import STANDING, radar_height, radar_median_height, radar_offaxis
from .utils import read_calibration

WALK_SPEED = 0.4        # m/s: radar horizontal speed of the person's track (feature v_horiz), over the whole window
WALK_WINDOW_S = 3.0     # s: trailing window
WALK_MIN_PATH = 1.0     # m: net horizontal displacement over the window
MOUNTING_CHECK_FILE = 'mounting_check.csv'    # in the mounting_check run (scripts/check_mounting_height.py)
DISTANCE_CURVE_FILE = 'distance_curve.csv'     # in the same run
DISTANCE_BINS = np.round(np.arange(0.5, 4.001, 0.25), 3)   # m from the radar axis: bin edges of the distance curve
MIN_BIN_FRAMES = 200      # walking frames a bin needs (over the persons the curve is fitted on)


def trailing_displacement(x, z, window):
    """(T,) net horizontal displacement (m) of the track (x, z) over the trailing `window` frames; positions carried
    forward over frames where the person isn't seen; NaN for the first `window` frames"""
    xs = pd.Series(np.asarray(x, float)).ffill().values
    zs = pd.Series(np.asarray(z, float)).ffill().values
    d = np.full(len(xs), np.nan)
    d[window:] = np.hypot(xs[window:] - xs[:-window], zs[window:] - zs[:-window])
    return d


def walking_frames(v_horiz, x, z, speed=WALK_SPEED, window_s=WALK_WINDOW_S, min_path=WALK_MIN_PATH, fps=FPS):
    """(T,) bool, causal: the person walks at this frame (standing and active by definition). From the radar's own
    track: speed v_horiz (m/s) above `speed` in every frame of the trailing window, and at least `min_path` of net
    displacement over it"""
    w = int(round(window_s * fps))
    fast = pd.Series((np.nan_to_num(np.asarray(v_horiz, float)) > speed).astype(float))
    fast = fast.rolling(w, min_periods=w).min().values > 0
    return fast & (trailing_displacement(x, z, w) >= min_path)


def session_walking(session, **kw):
    """walking_frames of a mobility.session.Session, inside the frames where the radar has data"""
    r = session.radar
    return walking_frames(r['v_horiz'], r['x_person'], r['z_person'], **kw) & ~np.isnan(r['p_sub'])


@dataclass(frozen=True)
class DistanceCurve:
    """how much the walking top-of-body height deviates, at a horizontal distance d from the radar axis, from the
    posture model's standing line, across persons: linear between the bin centres, constant beyond the outer ones"""
    centres: tuple        # m
    values: tuple         # m

    def __call__(self, d):
        return np.interp(np.nan_to_num(np.asarray(d, float)), self.centres, self.values)


def fit_distance_curve(offaxis, residual, person):
    """DistanceCurve from walking frames of several persons (per frame: distance from the axis, top-of-body residual,
    person): the median per DISTANCE_BINS bin of the residual minus the person's own median, so stature doesn't
    enter; bins with fewer than MIN_BIN_FRAMES frames are left out"""
    f = pd.DataFrame(dict(d=offaxis, r=residual, p=person)).dropna()
    f['r'] -= f.groupby('p').r.transform('median')
    b = pd.cut(f.d, DISTANCE_BINS, labels=False)
    g = f.groupby(b).r
    ok = g.size()[g.size() >= MIN_BIN_FRAMES].index.astype(int)
    centres = (DISTANCE_BINS[:-1] + DISTANCE_BINS[1:]) / 2
    return DistanceCurve(tuple(centres[ok]), tuple(g.median().loc[ok].values))


def walking_residuals(top, med, offaxis, walking, model, curve=None, with_offaxis=False):
    """(n, 2) [h_top, h_med] of the walking frames where both heights are seen, minus the standing mean of the posture
    model (mobility.posture.RadarPostureModel) at the same distance from the radar axis; with a DistanceCurve, the
    top-of-body residual is corrected by it too. with_offaxis: a third column, the distance from the axis."""
    w = np.asarray(walking, bool) & ~np.isnan(top) & ~np.isnan(med)
    d = np.nan_to_num(np.asarray(offaxis, float)[w])
    r = np.c_[np.asarray(top)[w], np.asarray(med)[w]] - (model.coef[STANDING, 0] + d[:, None] * model.coef[STANDING, 1])
    if curve is not None:
        r[:, 0] -= curve(d)
    return np.c_[r, d] if with_offaxis else r


def session_walking_residuals(session, model, curve=None, **kw):
    """walking_residuals of a Session, with the smoothed heights of the posture model"""
    return walking_residuals(radar_height(session), radar_median_height(session), radar_offaxis(session),
                             session_walking(session, **kw), model, curve)


@dataclass
class WalkingHeight:
    """Online accumulator of the walking height: feed it frames as they come (update), read the offset at any time.
    offset() = the median residual of the top-of-body height (and of the median height) over all walking frames so
    far, relative to the posture model's standing mean; the top-of-body one corrected by the distance curve, if given
    (distance_curves())."""
    model: object
    curve: DistanceCurve = None
    residuals: list = field(default_factory=list)

    def update(self, top, med, offaxis, walking):
        r = walking_residuals(top, med, offaxis, walking, self.model, self.curve)
        if len(r):
            self.residuals.append(r)

    @property
    def frames(self):
        return sum(len(r) for r in self.residuals)

    def seconds(self, fps=FPS):
        return self.frames / fps

    def offset(self):
        """(top, median) median residual (m); NaN before any walking frame"""
        if not self.residuals:
            return np.array([np.nan, np.nan])
        return np.median(np.concatenate(self.residuals), 0)


@dataclass(frozen=True)
class MountingCheck:
    """Is the radar's installed height plausible? An error of delta in the installed height moves every measured
    height by delta, so the resident's walking-height offset (top of the body, WalkingHeight.offset()[0]) moves by
    delta too. The check flags an offset farther than `limit` from the population's `reference`. The persons'
    statures spread the offsets as well, so the check catches only errors larger than that spread; `limit` and
    `reference` come from the other persons (scripts/check_mounting_height.py, for a given amount of walking).
    min_seconds: walking needed before the check says anything."""
    reference: float      # m: the population's median walking-height offset
    limit: float          # m: flag beyond reference +- limit
    min_seconds: float    # s of walking needed

    def estimate(self, offset):
        """the installed-height error the offset suggests (m; + = the radar is lower than configured, heights read
        high), stature not separated"""
        return offset - self.reference

    def flag(self, offset, seconds):
        """True: the offset is implausible (check the installed height); False: plausible; None: not enough walking"""
        if seconds < self.min_seconds or np.isnan(offset):
            return None
        return bool(abs(offset - self.reference) > self.limit)


def mounting_checks(path=None, data='all', correction='distance curve'):
    """{person: MountingCheck calibrated without that person, 'ALL': on every person}, for the amount of walking data
    `data` ('recording': one ~10-min recording, 'all': the person's recordings) and the distance correction of the
    offsets ('distance curve', with distance_curves(); 'none'), from the latest run of
    scripts/check_mounting_height.py (or `path`)"""
    d = read_calibration('mounting_check', MOUNTING_CHECK_FILE, path)
    d = d[(d.data == data) & (d.correction == correction)]
    return {r.person: MountingCheck(r.reference, r.limit, r.min_seconds) for r in d.itertuples()}


def distance_curves(path=None):
    """{person: DistanceCurve fitted on the other persons' walking frames, 'ALL': on every person}, from the latest
    run of scripts/check_mounting_height.py (or `path`)"""
    d = read_calibration('mounting_check', DISTANCE_CURVE_FILE, path)
    return {p: DistanceCurve(tuple(g.d), tuple(g.value)) for p, g in d.sort_values('d').groupby('person')}
