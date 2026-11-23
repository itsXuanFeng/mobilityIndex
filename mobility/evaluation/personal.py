"""
Personal calibration on top of leave-one-subject-out (docs/calibration.md), three setups:
    A  LOSO                              everything calibrated on the other persons, as in Categories 1-3
    B  LOSO + clip                       the person and room parameters adjusted on a short labelled clip of the
                                         evaluated recording (simulated installation session; labels only, no MoCap)
    C  LOSO + other recordings + clip    the person parameters from the person's other recordings (MoCap as the
                                         reference), the room parameters from the clip

The clip: the first occurrence of each session step in the recording (CLIP_LABELS): walking (standing posture,
active), eating on the chair (sitting, inactive), lying still in bed and pressing the wearable button (lying), and the
first sitting on the edge of the bed (sitting; between its label and the next, CLIP_GAP_LABELS). Each interval carries
its posture. For Category 3 heights the standing level uses only walking epochs within CLIP_STAND_OFFAXIS_M of the radar
axis (the session's "stand under the radar"). All setups are evaluated on the rest of the recording: the clip, the
postural-change labels right before and after a clip interval (their before or after posture is the clip's level), and
CLIP_MARGIN_S around both are left out.

Personal adjustments are shifts from a base value. A clip estimate compares the person's clip with the clips of the
reference persons (median radar level per activity class, median radar height per posture) and moves the threshold by
the mean shift of the two classes it separates. The shift is shrunk: base + w x shift, with w (SHRINK_GRID) chosen per
left-out person on the other persons, by simulating the clip on them.

Personal parameters:
    Category 1   radar active threshold     B: LOSO + w x clip shift (vs other persons' clips)
                                            C: LOSO + w_own x (person's other recordings, balanced accuracy against
                                               MoCap - LOSO), + w x clip shift (vs the person's own other clips)
                 energy conversion factor   C: LOSO + w_own x (the person's other recordings - LOSO); A and B: LOSO
                                               (so B's energy equals A's: the clip has no energy reference)
    Category 2   radar entry                the Category 1 radar threshold of the setup
                 radar exit delta, confirmation eps   C: the person's other recordings (as loso_radar_untied_levels)
    Category 3   h_stand                    B: LOSO + clip shift; C: LOSO + w x (the person's other recordings against
                                               MoCap - LOSO), w calibrated as the other weights
                 h_sit, h_floor             B and C: LOSO + clip shift
Everything else is the LOSO value of the Category 1-3 runs (thresholds.csv, category2/rules.csv,
category3/radar_posture_rule.csv).
"""
import numpy as np
import pandas as pd

from ..bouts import BoutRule, level_bouts, match_bouts
from ..config import EPOCH_FRAMES, EPOCH_S, LABEL_MARGIN_S
from ..io.recordings import parse_recording
from ..labels import ACTIVE, INACTIVE, time_classes
from ..metrics.category2 import set_bout_levels
from ..posture import (LYING, POSTURES, SITTING, STANDING, UNKNOWN, RadarPostureRule, calibrate_radar_thresholds,
                       epoch_posture, label_postures, label_transition, radar_height, radar_offaxis)
from ..session import load_session
from ..states import States
from ..transitions import in_window, mocap_transitions
from ..utils import best_threshold, epoch_mean
from .agreement import metric_rows, paired_states
from .category2 import RADAR_CONFIRM_EPS, RADAR_EXIT_DELTAS, bout_events, bout_frame, bout_scores
from .category3 import posture_scores, radar_events, transition_scores

SETUPS = ('A', 'B', 'C')
SETUP_NAMES = {'A': 'LOSO', 'B': 'LOSO + clip', 'C': 'LOSO + other recordings + clip'}
CLIP_LABELS = ('walk inside the main room', 'wander around in the room',          # walking: standing, active
               'start performing eating movements',                                # chair: sitting, inactive
               'lie on your back again and lie still', 'press the button on the wearable')   # bed: lying
CLIP_POSTURE = {'walk inside the main room': STANDING, 'wander around in the room': STANDING,
                'start performing eating movements': SITTING,
                'lie on your back again and lie still': LYING, 'press the button on the wearable': LYING}
CLIP_GAP_LABELS = {'sit down on the edge of the bed': SITTING}   # the gap after the label, until the next one
CLIP_STAND_OFFAXIS_M = 2.0    # Category 3 standing level: walking within this of the radar axis (radar estimate);
#                               2 m keeps >= 3 standing epochs in 96% of the recordings (1.5 m: 80%), height error <= ~11 cm
CLIP_MARGIN_S = LABEL_MARGIN_S
SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CLIP_POSTURES = (STANDING, SITTING, LYING)


# ---- the clip ----

def clip_intervals(labels):
    """[(start, end, posture)] of the clip: the first occurrence of each label in CLIP_LABELS, and the gap after the
    first occurrence of each label in CLIP_GAP_LABELS until the next label; and [(start, end)] of the postural-change
    labels right before and after each clip interval"""
    lb = labels.sort_values('start').reset_index(drop=True)
    norm = lb.label.str.strip().str.rstrip('.').str.lower()
    out, rows = [], []
    for name in CLIP_LABELS:
        i = norm.index[norm == name]
        if len(i):
            out.append((lb.start[i[0]], lb.end[i[0]], CLIP_POSTURE[name]))
            rows.append((i[0], i[0]))
    for name, posture in CLIP_GAP_LABELS.items():
        i = norm.index[norm == name]
        if len(i) and i[0] + 1 < len(lb):
            out.append((lb.end[i[0]], lb.start[i[0] + 1], posture))
            rows.append((i[0] + 1, i[0]))          # the gap lies between label i and label i + 1
    adjacent = set()
    for first, last in rows:
        for j in (first - 1, last + 1):
            if 0 <= j < len(lb):
                tr = label_transition(lb.label[j])
                if tr and tr[0] != tr[1]:
                    adjacent.add(j)
    return out, [(lb.start[j], lb.end[j]) for j in sorted(adjacent)]


def clip_masks(t, labels, margin=CLIP_MARGIN_S):
    """(clip, excluded, posture) per time t: inside a clip interval; within `margin` of one or of a postural-change
    label next to one (left out of the evaluation); the clip interval's posture (UNKNOWN outside the clip)"""
    t = np.asarray(t, float)
    clip, excl = np.zeros(len(t), bool), np.zeros(len(t), bool)
    posture = np.full(len(t), UNKNOWN)
    intervals, adjacent = clip_intervals(labels)
    for s, e, p in intervals:
        m = (t >= s) & (t <= e)
        clip |= m
        posture[m] = p
        excl |= (t >= s - margin) & (t <= e + margin)
    for s, e in adjacent:
        excl |= (t >= s - margin) & (t <= e + margin)
    return clip, excl, posture


# ---- one recording ----

def _fit(x, E, fill):
    x = np.asarray(x)[:E]
    return np.r_[x, np.full(E - len(x), fill, x.dtype)] if len(x) < E else x


def prepare(th, mocap_rule):
    """one recording (a row of thresholds.csv) for the three setups: per epoch the radar and MoCap levels and
    energies, validity, clip and exclusion masks, label class and scripted posture; per frame the radar height and
    distance from the axis, the window, the evaluation window (window minus the excluded clip); the MoCap transitions
    and posture per epoch under the person's MoCap posture rule"""
    s = load_session(*parse_recording(th.rec))
    rs, ms = paired_states(s, th.radar_threshold, th.mocap_threshold, active_feature=th.active_feature,
                           energy_feature=th.energy_feature)
    E = len(rs.t)
    mid = rs.t + EPOCH_S / 2
    clip, excl, clip_posture = clip_masks(mid, s.labels)
    # the clip needs labels and radar only, not MoCap (which often starts after the first labels): inside the script,
    # no caregiver, radar data
    radar_ok = _fit(epoch_mean(s.masks['in_script'] & s.masks['no_caregiver'] & ~np.isnan(s.radar['p_sub']),
                               EPOCH_FRAMES) >= 0.5, E, False)
    _, excl_f, _ = clip_masks(s.t, s.labels)
    off = radar_offaxis(s)
    window_eval = s.masks['window'] & ~excl_f
    mocap_ev = in_window(mocap_transitions(s, mocap_rule), s.t, window_eval)
    return dict(rec=th.rec, person=th.person, t_epoch=rs.t, radar_level=rs.level, radar_energy=rs.energy,
                mocap_level=ms.level, mocap_energy=ms.energy, valid=rs.valid, clip=clip & radar_ok,
                eval=rs.valid & ~excl, label_class=time_classes(mid, s.labels),
                script_posture=label_postures(mid, s.labels), clip_posture=clip_posture,
                offaxis_epoch=_fit(epoch_mean(off, EPOCH_FRAMES), E, np.nan), t=s.t, window_eval=window_eval,
                height=radar_height(s), offaxis=off, speeds=dict(v_peak=s.radar['v_peak']),
                mocap_events=mocap_ev,
                mocap_posture=_fit(epoch_posture(mocap_rule.classify(s.mocap), EPOCH_FRAMES), E, UNKNOWN))


def height_epochs(d, k):
    """corrected radar height per epoch (off-axis correction k)"""
    E = len(d['t_epoch'])
    return _fit(epoch_mean(d['height'] + k * np.nan_to_num(d['offaxis']), EPOCH_FRAMES), E, np.nan)


# ---- clip statistics and shifts ----

def clip_levels(d):
    """median radar level of the clip's active and inactive (label class) epochs"""
    c = d['clip'] & ~np.isnan(d['radar_level'])
    return {k: np.median(d['radar_level'][c & (d['label_class'] == k)]) if (c & (d['label_class'] == k)).any()
            else np.nan for k in (ACTIVE, INACTIVE)}


def clip_heights(d, k):
    """median corrected radar height of the clip's epochs per posture (standing, sitting, lying), each epoch taking
    the posture of its clip interval; standing only within CLIP_STAND_OFFAXIS_M of the radar axis"""
    h = height_epochs(d, k)
    c = d['clip'] & ~np.isnan(h)
    near = d['offaxis_epoch'] < CLIP_STAND_OFFAXIS_M
    out = {}
    for p in CLIP_POSTURES:
        m = c & (d['clip_posture'] == p) & (near if p == STANDING else True)
        out[p] = np.median(h[m]) if m.any() else np.nan
    return out


def _ref(stats, keys):
    """median over reference recordings of each statistic"""
    return {k: np.nanmedian([s[k] for s in stats]) if stats else np.nan for k in keys}


def _diff(a, b):
    x = a - b
    return 0.0 if np.isnan(x) else x


def level_shift(own, ref):
    """shift of the active threshold: mean shift of the active and inactive clip levels from the reference"""
    return (_diff(own[ACTIVE], ref[ACTIVE]) + _diff(own[INACTIVE], ref[INACTIVE])) / 2


def height_shifts(own, ref):
    """shifts of (h_stand, h_sit, h_floor): mean shift of the two postures each one separates (the floor doesn't
    move with the person)"""
    s = {p: _diff(own[p], ref[p]) for p in CLIP_POSTURES}
    return (s[STANDING] + s[SITTING]) / 2, (s[SITTING] + s[LYING]) / 2, s[LYING] / 2


# ---- Category 1 ----

def person_threshold(recs, mocap_thr):
    """radar active threshold maximising balanced accuracy against MoCap active over the recordings' valid epochs"""
    x = np.concatenate([d['radar_level'][d['valid']] for d in recs])
    y = np.concatenate([d['mocap_level'][d['valid']] > mocap_thr for d in recs])
    return best_threshold(x, y)


def energy_factor(recs):
    """total MoCap energy / total radar energy over the recordings' evaluated epochs"""
    m = sum(np.nansum(d['mocap_energy'][d['eval']]) for d in recs)
    r = sum(np.nansum(d['radar_energy'][d['eval']]) for d in recs)
    return m / r


def active_time_error(d, thr, mocap_thr):
    """|radar - MoCap| active time (min) on the evaluated epochs"""
    e = d['eval']
    return abs(((d['radar_level'][e] > thr).sum() - (d['mocap_level'][e] > mocap_thr).sum()) * EPOCH_S / 60)


# ---- Category 2 ----

def person_bout_levels(recs, entry, mocap_entry, mocap_factors, rule, deltas=RADAR_EXIT_DELTAS,
                       eps=RADAR_CONFIRM_EPS):
    """(delta, eps) on the recordings' valid epochs, as loso_radar_untied_levels (the definition until 2026-11-21; the
    LOSO reference is now one margin kappa, read from category2/rules.csv as delta = eps = kappa): eps by the bout
    count, then delta by the mean bout duration; ties: the smallest"""
    def mean_duration(b):
        return (b[:, 1] - b[:, 0]).mean() if len(b) else np.nan

    mb = [level_bouts(d['mocap_level'], d['valid'], mocap_entry, mocap_factors[0] * mocap_entry,
                      mocap_factors[1] * mocap_entry, rule) for d in recs]
    count_err, dur_err = {}, {}
    for e in eps:
        for dl in deltas:
            rb = [level_bouts(d['radar_level'], d['valid'], entry, entry - dl, entry + e, rule) for d in recs]
            count_err[(dl, e)] = sum(abs(len(r) - len(m)) for r, m in zip(rb, mb))
            dur_err[(dl, e)] = np.nanmean([abs(mean_duration(r) - mean_duration(m)) for r, m in zip(rb, mb)])
    e_best = min(count_err, key=lambda k: (count_err[k], k[1], k[0]))[1]
    d_best = min(deltas, key=lambda dl: (round(dur_err[(dl, e_best)], 9), dl))
    return d_best, e_best


def states_pair(d, thr, mocap_thr):
    """radar and MoCap States of a prepared recording on its evaluated epochs"""
    v = d['eval']
    rs = States('radar', d['t_epoch'], d['radar_energy'], d['radar_level'], d['radar_level'] > thr, v.copy())
    ms = States('mocap', d['t_epoch'], d['mocap_energy'], d['mocap_level'], d['mocap_level'] > mocap_thr, v.copy())
    return rs, ms


def bout_results(d, thr, mocap_thr, delta, eps, mocap_factors, rule):
    """Category 2 metric rows and bout scores on the evaluated epochs"""
    rs, ms = states_pair(d, thr, mocap_thr)
    rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
    set_bout_levels(rs, ms, (thr - delta, thr + eps), (mocap_factors[0] * mocap_thr, mocap_factors[1] * mocap_thr))
    frame = bout_frame(rs, ms)
    return metric_rows(d['rec'], rs, ms, 2), bout_scores(frame, bout_events(frame))


# ---- Category 3 ----

def posture_results(d, rule):
    """posture per epoch scores, transition scores and posture times (radar, MoCap) on the evaluated epochs"""
    frames = rule.classify(d['height'], d['offaxis'])
    E = len(d['t_epoch'])
    rp = _fit(epoch_posture(frames, EPOCH_FRAMES), E, UNKNOWN)
    mp, v = d['mocap_posture'], d['eval']
    ps = posture_scores(rp, mp, height_epochs(d, rule.offaxis_k), v)
    dd = dict(t=d['t'], window=d['window_eval'], speeds=d['speeds'])
    ts = transition_scores(radar_events(dd, rule, frames), d['mocap_events'])
    times = {f'{n}_time': ((v & (rp == c)).sum() / 60, (v & (mp == c)).sum() / 60) for c, n in enumerate(POSTURES)}
    return ps, ts, times


def shifted_rule(base, shifts, w, h_stand=None):
    """base RadarPostureRule moved by w x shifts (h_stand, h_sit, h_floor); h_stand: replaces the standing
    threshold (setup C: the shrunk value from the person's other recordings) and is not shifted"""
    hs = base.h_stand + w * shifts[0] if h_stand is None else h_stand
    return RadarPostureRule(hs, base.h_sit + w * shifts[1], base.h_floor + w * shifts[2], base.min_hold_s,
                            base.offaxis_k)


def person_h_stand(recs, k):
    """h_stand calibrated on the recordings against their MoCap posture (balanced accuracy, standing | sitting)"""
    ep = []
    for d in recs:
        h, keep = height_epochs(d, k), d['valid'] & (d['mocap_posture'] != UNKNOWN)
        ep.append(pd.DataFrame(dict(person=d['person'], height=h, posture=d['mocap_posture']))[keep])
    return calibrate_radar_thresholds(pd.concat(ep, ignore_index=True))[0]
