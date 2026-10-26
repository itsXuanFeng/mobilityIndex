"""
Postural transitions (Category 3) as events, from MoCap and from the radar, in the same form: one row per transition
with its time, kind, direction and peak radial speed.

MoCap: the changes of the posture sequence (mobility.posture), once each posture has lasted min_hold_s.

Radar, main method (with the angle): the same, on the radar posture sequence: the posture from the height of the top of
the body (mobility.posture.RadarPostureRule), carried forward while the person isn't seen, transitions where it changes
and holds (radar_posture_transitions). The same rule as for MoCap, so the two define a transition the same way.

Radar, comparison (range and Doppler only, the first version; with or without the angle features), in four steps:
    1. bursts      runs of vertical motion of one sign: the signed Doppler centroid, smoothed over 0.5 s, beyond V_ON;
                   same-sign runs closer than MERGE_GAP_S merged. + = away from the radar = going down.
    2. features    per burst: direction, duration, peak speed, the range of the nearest moving reflector before and
                   after (~ the top of the body), the range change in the burst's direction, and how much moves just
                   before and after (a seated or lying person is mostly still).
    3. kind        a classifier on these features (BurstModel), trained against the MoCap transitions of the *other*
                   persons. Without the angle, the slant range mixes height with the horizontal distance: a person
                   standing 2 m off-axis is as far from the radar as one lying below it. No hand rule on range and
                   Doppler separated transitions from walking, bending and moving in bed (docs/category3.md), so the
                   rule is learned, leaving the person out.
    4. events      bursts that are a transition with probability >= min_prob, one per direction within NMS_S (the
                   most probable). min_prob is calibrated like a threshold: on out-of-fold classifications of the
                   other persons (calibrate_min_prob).

Peak speed: max |radial speed| within PEAK_HALF_S of the event time. MoCap: the fastest joint along the line to the
radar (signal vr_max). Radar, main method: a power-weighted quantile of the person's moving speeds (angle features
v_q900 ... v_q995, the quantile calibrated against MoCap in evaluate_category3.py); the burst classifiers keep the
fastest strong cell (feature v_peak), which reads high because of Doppler leakage.
"""
import numpy as np
import pandas as pd

from .config import FPS, RADAR_XYZ
from .posture import FLOOR, LEVEL, LYING, SITTING, STANDING, UNKNOWN, label_transitions, radar_height
from .utils import log_level, nan_smooth

SHORT = {STANDING: 'stand', SITTING: 'sit', LYING: 'lie'}
COUNTED = ('sit_to_stand', 'stand_to_sit', 'lie_to_sit')     # the transitions counted by the Category 3 metrics
NONE = 'none'

MATCH_TOL_S = 1.0       # radar and MoCap events match when they are this close in time (SCOPE.md: +-1 s)
PEAK_HALF_S = 1.0       # peak speed: within this of the event time

# radar bursts
V_ON = 0.05             # m/s: smoothed signed Doppler centroid beyond this is directed motion (still: ~0)
SMOOTH_S = 0.5
MERGE_GAP_S = 0.5
MIN_BURST_S = 0.3
EDGE_S = 0.5            # range before / after a burst: median of the nearest moving reflector within this of its edge
CONTEXT_S = 2.5         # how much moves before / after a burst: over this long
NMS_S = 1.5             # one event per direction within this


def kind(frm, to):
    """'sit_to_stand', ...; any change to or from the floor is 'to_floor' / 'from_floor'"""
    if FLOOR in (frm, to):
        return 'to_floor' if to == FLOOR else 'from_floor'
    return f'{SHORT[frm]}_to_{SHORT[to]}'


def direction(frm, to):
    """+1 down (away from the radar, + Doppler), -1 up"""
    return 1 if LEVEL[to] < LEVEL[frm] else -1


EVENT_COLUMNS = ['t', 'start', 'end', 'frm', 'to', 'kind', 'direction', 'peak_v']


# ---- MoCap: from a posture sequence ----

def stable_segments(posture, hold):
    """(start, end, posture) of the runs of one posture at least `hold` samples long, in time order"""
    p = np.asarray(posture)
    edges = np.flatnonzero(np.diff(p)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(p)]
    keep = (ends - starts >= hold) & (p[starts] != UNKNOWN)
    return list(zip(starts[keep], ends[keep], p[starts[keep]]))


def posture_transitions(t, posture, min_hold_s, fps=FPS):
    """events between consecutive stable postures that differ, when nothing unknown lies between them. The event
    spans the gap between the two stable postures; its time is the middle of the gap."""
    t = np.asarray(t, float)
    seg = stable_segments(posture, max(int(round(min_hold_s * fps)), 1))
    rows = []
    for (s0, e0, a), (s1, e1, b) in zip(seg, seg[1:]):
        if a != b and not (np.asarray(posture[e0:s1]) == UNKNOWN).any():
            rows.append(dict(t=(t[e0 - 1] + t[s1]) / 2, start=t[e0 - 1], end=t[s1], frm=int(a), to=int(b),
                             kind=kind(a, b), direction=direction(a, b), peak_v=np.nan))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def mocap_transitions(session, rule):
    """the MoCap transitions of a session (on the radar frames), with their peak radial speed and offaxis_m, the
    horizontal distance of the centre of gravity from the point below the radar"""
    ev = add_peak(posture_transitions(session.t, rule.classify(session.mocap), rule.min_hold_s), session.t,
                  session.mocap['vr_max'])
    i = np.clip(np.searchsorted(session.t, ev.t.values), 0, len(session.t) - 1)
    return ev.assign(offaxis_m=np.hypot(session.mocap['cog_x'][i] - RADAR_XYZ[0],
                                        session.mocap['cog_z'][i] - RADAR_XYZ[2]))


def radar_posture_transitions(t, posture, v_peak, min_hold_s):
    """the radar transitions: changes of the radar posture per frame (RadarPostureRule.classify of radar_height),
    held >= min_hold_s, with the radar peak radial speed (feature v_peak)"""
    return add_peak(posture_transitions(t, posture, min_hold_s), t, v_peak)


def add_peak(events, t, speed, half=PEAK_HALF_S):
    """peak_v = max |speed| within `half` of each event time"""
    peaks = []
    for x in events.t:
        m = (t >= x - half) & (t <= x + half)
        v = np.abs(speed[m])
        peaks.append(np.nanmax(v) if m.any() and not np.isnan(v).all() else np.nan)
    return events.assign(peak_v=peaks)


# ---- radar: bursts and their features ----

def _median(x, i0, i1):
    x = x[max(i0, 0):max(min(i1, len(x)), 0)]
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else np.nan


def _mean(x, i0, i1):
    x = x[max(i0, 0):max(min(i1, len(x)), 0)]
    return float(np.nanmean(x)) if len(x) and not np.isnan(x).all() else np.nan


def _runs(mask):
    m = np.concatenate([[False], mask, [False]])
    return np.flatnonzero(np.diff(m.astype(np.int8))).reshape(-1, 2)


BURST_FEATURES = ['direction', 'duration', 'v_smooth', 'r0', 'r1', 'rise', 'r_pre', 'r_post', 'd_pre', 'd_post',
                  'moving_pre', 'moving_post', 'level_in', 'level_pre', 'level_post', 'since_prev', 'until_next']
# with the angle: the top-of-body height before / after instead of only the slant range, and where the person is
ANGLE_BURST_FEATURES = BURST_FEATURES + ['h0', 'h1', 'h_rise', 'h_pre', 'h_post', 'offaxis']


def bursts(session):
    """
    every burst of directed vertical motion in the recording:
        t, start, end       middle, first and last frame time (epoch s)
        direction           +1 away from the radar (down), -1 towards it (up)
        duration            s
        v_smooth            peak |smoothed Doppler centroid| (m/s)
        r0, r1              range of the nearest moving reflector at the start and end (m)
        rise                range change in the burst's direction, (r1 - r0) * direction: > 0 when range and Doppler
                            agree, as for a vertical move
        r_pre, r_post       that range over CONTEXT_S before and after; d_pre, d_post = |r_pre - r0|, |r_post - r1|
        moving_pre/_post    share of frames with a clearly moving reflector over CONTEXT_S before and after
        level_in/_pre/_post activity level (log n_cells) in, before and after the burst
        since_prev, until_next   s to the neighbouring bursts (either direction)
        peak_v              radar peak radial speed (v_peak) within PEAK_HALF_S of t
    with the angle features in the session (angle pass), also:
        h0, h1, h_pre, h_post    top-of-body height (m, radar_height) at the start and end, and before and after
        h_rise              height change in the burst's direction, (h0 - h1) * direction: > 0 for a vertical move
        offaxis             horizontal distance of the person from the point below the radar (m)
    """
    t, r = session.t, session.radar['r_near']
    v = np.nan_to_num(nan_smooth(session.radar['v_centroid'], int(SMOOTH_S * FPS)))
    level = log_level(session.radar['n_cells'])
    moving = (~np.isnan(r)).astype(float)
    moving[np.isnan(session.radar['n_cells'])] = np.nan
    edge, ctx = int(EDGE_S * FPS), int(CONTEXT_S * FPS)
    rows = []
    for sign in (1, -1):
        merged = []
        for a, b in _runs(sign * v > V_ON):
            if merged and a - merged[-1][1] <= MERGE_GAP_S * FPS:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a < MIN_BURST_S * FPS:
                continue
            r0, r1 = _median(r, a - edge, a + edge), _median(r, b - edge, b + edge)
            r_pre, r_post = _median(r, a - edge - ctx, a - edge), _median(r, b + edge, b + edge + ctx)
            rows.append(dict(t=(t[a] + t[b - 1]) / 2, start=t[a], end=t[b - 1], direction=sign,
                             duration=(b - a) / FPS, v_smooth=float(np.abs(v[a:b]).max()), r0=r0, r1=r1,
                             rise=(r1 - r0) * sign, r_pre=r_pre, r_post=r_post, d_pre=abs(r_pre - r0),
                             d_post=abs(r_post - r1), moving_pre=_mean(moving, a - ctx, a),
                             moving_post=_mean(moving, b, b + ctx), level_in=_mean(level, a, b),
                             level_pre=_mean(level, a - ctx, a), level_post=_mean(level, b, b + ctx)))
    b = pd.DataFrame(rows, columns=['t', 'start', 'end'] + BURST_FEATURES[:-2]).sort_values('t', ignore_index=True)
    b['since_prev'] = (b.start - b.end.shift(1)).fillna(60.0).clip(upper=60)
    b['until_next'] = (b.start.shift(-1) - b.end).fillna(60.0).clip(upper=60)
    if 'h_top' in session.radar:
        h = radar_height(session)
        off = np.hypot(session.radar['x_person'] - RADAR_XYZ[0], session.radar['z_person'] - RADAR_XYZ[2])
        i = np.clip(np.searchsorted(t, b.start.values), 0, len(t) - 1)
        j = np.clip(np.searchsorted(t, b.end.values), 0, len(t) - 1)
        b['h0'] = [_median(h, a - edge, a + edge) for a in i]
        b['h1'] = [_median(h, e - edge, e + edge) for e in j]
        b['h_rise'] = (b.h0 - b.h1) * b.direction
        b['h_pre'] = [_median(h, a - edge - ctx, a - edge) for a in i]
        b['h_post'] = [_median(h, e + edge, e + edge + ctx) for e in j]
        b['offaxis'] = [_median(off, a, e + 1) for a, e in zip(i, j)]
    return add_peak(b, t, session.radar['v_peak'])


def burst_targets(b, mocap_events, tol=MATCH_TOL_S):
    """training target per burst: the kind of the nearest MoCap transition of the same direction within tol, or
    'none'"""
    out = np.full(len(b), NONE, object)
    for i, (x, d) in enumerate(zip(b.t.values, b.direction.values)):
        e = mocap_events[(mocap_events.direction == d) & ((mocap_events.t - x).abs() <= tol)]
        if len(e):
            out[i] = e.kind.values[np.argmin(np.abs(e.t.values - x))]
    return out


MIN_CLASS_BURSTS = 10   # kinds with fewer training bursts are left out of the training (stand_to_lie, lie_to_stand)


class BurstModel:
    """gradient-boosted trees: burst features -> transition kind or 'none'. Early stopping on a held-back 10% of the
    training bursts: without it the trees run to saturated probabilities (0 or 1) that fit the training persons and
    miss whole held-out persons."""

    def __init__(self, features=BURST_FEATURES, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.features = list(features)
        self.model = HistGradientBoostingClassifier(**{**dict(max_iter=200, learning_rate=0.1, early_stopping=True,
                                                            validation_fraction=0.1, n_iter_no_change=10,
                                                            random_state=0), **kw})

    def fit(self, b, targets):
        targets = np.asarray(targets, object)
        kinds, n = np.unique(targets, return_counts=True)
        ok = np.isin(targets, kinds[n >= MIN_CLASS_BURSTS])
        self.model.fit(b[self.features].values[ok], targets[ok])
        return self

    def classify(self, b):
        """b plus, per burst, `kind`: the most probable transition kind, and `prob`: the probability that it is a
        transition at all (1 - P(none))"""
        if not len(b):
            return b.assign(kind=pd.Series(dtype=object), prob=pd.Series(dtype=float))
        p = self.model.predict_proba(b[self.features].values)
        c = list(self.model.classes_)
        none = p[:, c.index(NONE)] if NONE in c else np.zeros(len(b))
        p = p.copy()
        if NONE in c:
            p[:, c.index(NONE)] = -1
        return b.assign(kind=np.asarray(c, object)[p.argmax(1)], prob=1 - none)


def radar_transitions(classified, min_prob, nms_s=NMS_S):
    """events from the classified bursts of one recording (BurstModel.classify): bursts that are a transition with
    probability >= min_prob, one per direction within nms_s (the most probable)"""
    cand = classified[classified.prob >= min_prob].sort_values('prob', ascending=False)
    kept = []
    for x in cand.itertuples():
        if not any(abs(x.t - y.t) < nms_s and x.direction == y.direction for y in kept):
            kept.append(x)
    ev = pd.DataFrame([dict(t=x.t, start=x.start, end=x.end, frm=np.nan, to=np.nan, kind=x.kind,
                            direction=x.direction, peak_v=x.peak_v, prob=x.prob) for x in kept],
                      columns=EVENT_COLUMNS + ['prob'])
    return ev.sort_values('t', ignore_index=True)


GRID_MIN_PROB = tuple(np.round(np.arange(0.05, 0.71, 0.05), 2))


def calibrate_min_prob(recordings, grid=GRID_MIN_PROB, tol=MATCH_TOL_S):
    """the detection threshold with the best pooled F1 (same kind, within tol) over recordings: a list of
    (classified bursts, MoCap events, keep) with keep(events) -> the events inside the evaluation window. The
    classifications must be out of fold: from models that never saw the recording's person."""
    best, best_f1 = grid[0], -1.0
    for g in grid:
        hit = n = 0
        for classified, mocap, keep in recordings:
            radar = keep(radar_transitions(classified, g))
            hit += len(match_events(mocap, radar, tol))
            n += len(mocap) + len(radar)
        f1 = 2 * hit / n if n else 0.0
        if f1 > best_f1:
            best, best_f1 = g, f1
    return float(best), best_f1


# ---- matching ----

def match_events(ref, est, tol=MATCH_TOL_S, typed=True):
    """one-to-one matching of two event tables by time (closest pairs first), within tol and, when typed, of the
    same kind. Returns {index in ref: index in est} (positional)."""
    if not len(ref) or not len(est):
        return {}
    d = np.abs(ref.t.values[:, None] - est.t.values[None, :])
    if typed:
        d[ref.kind.values[:, None] != est.kind.values[None, :]] = np.inf
    out, used = {}, set()
    for k in np.argsort(d, axis=None):
        i, j = np.unravel_index(k, d.shape)
        if d[i, j] > tol:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def scripted_transitions(labels):
    """the scripted postural changes as events: label, start, end, kind, direction; t = middle of the label"""
    lt = label_transitions(labels)
    return lt.assign(t=(lt.start + lt.end) / 2, kind=[kind(a, b) for a, b in zip(lt.frm, lt.to)],
                     direction=[direction(a, b) for a, b in zip(lt.frm, lt.to)])


def match_scripted(scripted, events, tol=MATCH_TOL_S):
    """one-to-one: each scripted change gets the event of its kind closest to the middle of the label, within the
    label +- tol. Labels are set by hand and also hold the still moments around the change, so the label span is
    the tolerance, not its middle. Returns {index in scripted: index in events} (positional)."""
    out, used = {}, set()
    for i, s in enumerate(scripted.itertuples()):
        ok = ((events.kind == s.kind) & (events.t >= s.start - tol) & (events.t <= s.end + tol)).values
        cand = [j for j in np.argsort(np.abs(events.t.values - s.t)) if ok[j] and j not in used]
        if cand:
            out[i] = cand[0]
            used.add(cand[0])
    return out


def in_window(events, t, window):
    """the events whose time falls on a frame inside the window"""
    if not len(events):
        return events
    i = np.clip(np.searchsorted(t, events.t.values), 0, len(t) - 1)
    return events[np.asarray(window)[i]].reset_index(drop=True)
