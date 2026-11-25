"""
Category 3: postural transitions per event and posture per epoch, MoCap as the reference, and everything the radar
calibrates for them leaving the evaluated person out (docs/category3.md):
    posture model      emission, evidence weight and change probability of mobility.posture.RadarPostureModel, the
                       main method (calibrate_posture_model)
    height rule        thresholds, off-axis correction and hold of mobility.posture.RadarPostureRule, a comparison
                       (calibrate_height_rule)
    peak Doppler       which power-weighted quantile of the person's speeds (calibrate_peak)
    comparisons        burst classifiers (burst_classifier_fold) and the hand rule (hand_rule_baseline)

These work on prepared recordings (prepare_recording): one dict per recording, `data` = {recording name: dict}.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from ..config import EPOCH_FRAMES, FPS
from ..io.recordings import parse_recording
from ..posture import (EMISSION_STEP, FLOOR, GRID_BETA, GRID_MIN_HOLD_S, GRID_OFFAXIS_K, GRID_P_MOVE, LYING, POSTURES,
                       SITTING, STANDING, UNKNOWN, RadarPostureModel, RadarPostureRule, calibrate_radar_thresholds,
                       decode_postures, epoch_posture, fit_posture_emission, radar_height, radar_median_height,
                       radar_moving, radar_offaxis)
from ..radar.angle import PEAK_FEATURES
from ..session import load_session
from ..transitions import (ANGLE_BURST_FEATURES, BURST_FEATURES, COUNTED, MATCH_TOL_S, NONE, BurstModel, add_peak,
                           burst_targets, bursts, calibrate_min_prob, hand_rule_events, in_window, match_events,
                           mocap_transitions, radar_posture_transitions, radar_transitions)
from ..utils import epoch_mean, fold_of

INNER_FOLDS = 5    # classifier detection threshold: out-of-fold classifications of the training persons
CLASSIFIERS = {'classifier': BURST_FEATURES, 'classifier+angle': ANGLE_BURST_FEATURES}
# hand-rule baseline: a burst is a transition when fast, long and its range change agrees with its Doppler sign
HAND_RULE_GRID = list(itertools.product((0.1, 0.15, 0.2, 0.25, 0.3), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                                        (0.3, 0.5, 0.8, 1.0)))     # v_smooth (m/s), rise (m), duration (s)


# ---- scores of one recording ----

def transition_scores(radar, mocap, tol=MATCH_TOL_S):
    """one recording, MoCap as the reference, events inside the evaluation window. Typed: matched within tol and of
    the same kind. Any: matched within tol whatever the kind (a transition is detected, its kind may be wrong).
    Matched pairs (typed): time error and peak-speed error (radar - MoCap)."""
    n_m, n_r = len(mocap), len(radar)
    out = dict(mocap_events=n_m, radar_events=n_r)
    for name, typed in (('', True), ('_any', False)):
        hit = len(match_events(mocap, radar, tol, typed))
        out.update({f'matched{name}': hit, f'precision{name}': hit / n_r if n_r else np.nan,
                    f'recall{name}': hit / n_m if n_m else np.nan,
                    f'f1{name}': 2 * hit / (n_m + n_r) if n_m + n_r else np.nan})
    m = match_events(mocap, radar, tol)
    if m:
        i, j = np.array(list(m.keys())), np.array(list(m.values()))
        out.update(time_mae_s=np.abs(radar.t.values[j] - mocap.t.values[i]).mean(),
                   peak_mae=np.nanmean(np.abs(radar.peak_v.values[j] - mocap.peak_v.values[i])),
                   peak_bias=np.nanmean(radar.peak_v.values[j] - mocap.peak_v.values[i]))
    return out


def transition_events(radar, mocap, t0, tol=MATCH_TOL_S):
    """every event of both sensors in one recording: time (s since t0), kind, direction, peak speed; whether it is
    matched (typed), and the kind of the other sensor's event matched in time whatever the kind ('' if none); for
    MoCap events, the distance from the radar axis (offaxis_m)"""
    typed, any_ = match_events(mocap, radar, tol), match_events(mocap, radar, tol, typed=False)
    rows = []
    for src, ev, other, m, m_any in (('mocap', mocap, radar, typed, any_),
                                     ('radar', radar, mocap, {j: i for i, j in typed.items()},
                                      {j: i for i, j in any_.items()})):
        for i, e in enumerate(ev.itertuples()):
            rows.append(dict(source=src, t_s=e.t - t0, kind=e.kind, direction=e.direction, peak_v=e.peak_v,
                             matched=i in m, other_kind=other.kind.values[m_any[i]] if i in m_any else '',
                             dt_s=(e.t - other.t.values[m_any[i]]) if i in m_any else np.nan,
                             offaxis_m=getattr(e, 'offaxis_m', np.nan)))
    return pd.DataFrame(rows, columns=['source', 't_s', 'kind', 'direction', 'peak_v', 'matched', 'other_kind',
                                       'dt_s', 'offaxis_m'])


def per_kind_table(events):
    """per transition kind, pooled over transition_events of all recordings: MoCap and radar transitions, matched,
    F1, and as_<kind>: the radar kind (by time only) of the MoCap transitions ('missed': none)"""
    rows = []
    for k in sorted(set(events.kind)):
        m, r = events[(events.source == 'mocap') & (events.kind == k)], events[(events.source == 'radar')
                                                                               & (events.kind == k)]
        rows.append(dict(kind=k, mocap=len(m), radar=len(r), matched=int(m.matched.sum()),
                         f1=2 * m.matched.sum() / (len(m) + len(r)) if len(m) + len(r) else np.nan,
                         **{f'as_{o or "missed"}': int((m.other_kind == o).sum()) for o in sorted(set(m.other_kind))}))
    return pd.DataFrame(rows).fillna(0)


def posture_scores(radar, mocap, height, valid):
    """one recording, per 1 s epoch, MoCap posture as the reference (epochs valid and with a known MoCap posture):
    F1 of the radar posture for each posture MoCap shows, their mean, and the AUC of the radar height for each pair
    of neighbouring postures (threshold-free: how well height separates them); seen = share of the epochs where the
    radar sees the person"""
    v = np.asarray(valid, bool) & (np.asarray(mocap) != UNKNOWN)
    r, m, h = np.asarray(radar)[v], np.asarray(mocap)[v], np.asarray(height, float)[v]
    out = dict(epochs=int(v.sum()), seen=float(np.mean(~np.isnan(h))) if v.any() else np.nan)
    f1s = []
    for k, name in enumerate(POSTURES):
        f = f1_score(m == k, r == k) if (m == k).any() else np.nan
        out[f'f1_{name}'] = f
        if not np.isnan(f):
            f1s.append(f)
    out['f1_mean'] = float(np.mean(f1s)) if f1s else np.nan
    for upper, lower in ((STANDING, SITTING), (SITTING, LYING), (LYING, FLOOR)):
        g = np.isin(m, [upper, lower]) & ~np.isnan(h)
        both = (m[g] == upper).any() and (m[g] == lower).any()
        out[f'auc_{POSTURES[upper]}_{POSTURES[lower]}'] = roc_auc_score(m[g] == upper, h[g]) if both else np.nan
    return out


# ---- prepared recordings ----

def prepare_recording(rec, mocap_rules, active):
    """
    one recording ('person_x/scenario_y') for the Category 3 evaluation, a dict of:
        rec, person
        t, window       frame times and the evaluation window per frame
        height, offaxis the radar's top-of-body height and distance from the axis per frame (smoothed)
        height_med      the median height of the person's points per frame (smoothed)
        moving          per frame: the person moves (Category 1 activity, active = (feature, threshold))
        speeds          per-frame radar speeds for the peak Doppler: v_peak and the quantiles PEAK_FEATURES
        valid_epoch     epochs inside the window
        bursts          the radar bursts (mobility.transitions.bursts), with in_window
        mocap           {MoCap posture rule: MoCap transitions inside the window}, for every rule in mocap_rules
        mocap_epoch     {MoCap posture rule: MoCap posture per epoch}
        mocap_frame     {MoCap posture rule: MoCap posture per frame (int8)}
    """
    s = load_session(*parse_recording(rec))
    w = s.masks['window']
    b = bursts(s)
    b['in_window'] = w[np.clip(np.searchsorted(s.t, b.t.values), 0, len(s.t) - 1)]
    h, off = radar_height(s), radar_offaxis(s)
    mocap, mocap_epoch, mocap_frame = {}, {}, {}
    for r in mocap_rules:
        mocap[r] = in_window(mocap_transitions(s, r), s.t, w)
        frames = r.classify(s.mocap)
        mocap_epoch[r] = epoch_posture(frames, EPOCH_FRAMES)
        mocap_frame[r] = frames.astype(np.int8)
    speeds = {k: s.radar[k] for k in ('v_peak',) + PEAK_FEATURES}
    return dict(rec=rec, person=s.person, t=s.t, window=w, height=h, offaxis=off, height_med=radar_median_height(s),
                moving=radar_moving(s, active[1], active[0]), speeds=speeds, valid_epoch=s.epoch_mask('window'),
                bursts=b, mocap=mocap, mocap_epoch=mocap_epoch, mocap_frame=mocap_frame)


def window_filter(d):
    """events -> the events of prepared recording d inside its evaluation window"""
    return lambda e: in_window(e, d['t'], d['window'])


def others(data, person):
    """the prepared recordings of every person but `person` (all when None)"""
    return [d for d in data.values() if d['person'] != person]


# ---- main method: the posture model ----

def model_inputs(d):
    """the per-frame inputs of RadarPostureModel for prepared recording d"""
    return d['height'], d['height_med'], d['offaxis'], d['moving']


def model_events(d, posture, speed='v_peak'):
    """the radar transitions of prepared recording d from its decoded posture per frame (no hold beyond the model's
    own minimum duration), inside its window; peak speed from the per-frame speed `speed`"""
    return window_filter(d)(radar_posture_transitions(d['t'], posture, d['speeds'][speed], 1.0 / FPS))


def fit_model_emission(recordings, mrule, step=EMISSION_STEP):
    """(coef, cov) of RadarPostureModel on the prepared recordings: every `step`-th frame inside the window where the
    radar sees the person and MoCap (under MoCap posture rule mrule) gives the posture"""
    cols = [[], [], [], []]
    for d in recordings:
        p = d['mocap_frame'][mrule]
        ok = d['window'] & ~np.isnan(d['height']) & ~np.isnan(d['height_med']) & (p != UNKNOWN)
        ok[np.arange(len(ok)) % step != 0] = False
        for c, x in zip(cols, (d['height'], d['height_med'], np.nan_to_num(d['offaxis']), p)):
            c.append(np.asarray(x)[ok])
    return fit_posture_emission(*(np.concatenate(c) for c in cols))


def calibrate_posture_model(data, rule_of, person):
    """(RadarPostureModel, training transition F1) calibrated on every person but `person` ('ALL' when None), with
    MoCap under that person's MoCap posture rule: the emission fitted on their frames, the minimum posture duration
    = that rule's min_hold_s, then beta and p_move with the best pooled transition F1 against MoCap. Ties: the first
    in the grid."""
    mrule = fold_of(rule_of, person)
    train = others(data, person)
    coef, cov = fit_model_emission(train, mrule)
    base = RadarPostureModel(coef, cov, d_min_s=mrule.min_hold_s)
    best = None
    for beta, p_move in itertools.product(GRID_BETA, GRID_P_MOVE):
        model = base.with_settings(beta, p_move)
        hit = n = 0
        for d, post in zip(train, decode_postures(model, [model_inputs(d) for d in train])):
            radar, mocap = model_events(d, post), d['mocap'][mrule]
            hit += len(match_events(mocap, radar))
            n += len(mocap) + len(radar)
        f1 = 2 * hit / n if n else 0.0
        if best is None or f1 > best[1]:
            best = (model, f1)
    return best


# ---- comparison: posture from height (the height rule) ----

def radar_events(d, rule, posture=None, speed='v_peak'):
    """the radar transitions of prepared recording d under a RadarPostureRule, inside its window; peak speed from the
    per-frame speed `speed`. posture: the rule's posture per frame, when already classified"""
    if posture is None:
        posture = rule.classify(d['height'], d['offaxis'])
    return window_filter(d)(radar_posture_transitions(d['t'], posture, d['speeds'][speed], rule.min_hold_s))


def height_epochs(d, rule):
    """the corrected radar height per epoch of prepared recording d"""
    return epoch_mean(rule.height(d['height'], d['offaxis']), EPOCH_FRAMES)


def calibrate_height_rule(data, rule_of, person):
    """(RadarPostureRule, training transition F1) calibrated on every person but `person` ('ALL' when None), with
    MoCap under that person's MoCap posture rule (rule_of: {person: MoCap PostureRule}): per off-axis correction,
    thresholds per epoch; then the correction and hold with the best transition F1. Ties: the smallest correction
    and hold."""
    mrule = fold_of(rule_of, person)
    train = others(data, person)
    best = None
    for k in GRID_OFFAXIS_K:
        ep = []
        for d in train:
            h, mp = height_epochs(d, RadarPostureRule(0, 0, 0, 0, k)), d['mocap_epoch'][mrule]
            E = min(len(h), len(mp))
            keep = d['valid_epoch'][:E] & (mp[:E] != UNKNOWN)
            ep.append(pd.DataFrame(dict(person=d['person'], height=h[:E], posture=mp[:E]))[keep])
        th = calibrate_radar_thresholds(pd.concat(ep, ignore_index=True))
        postures = {d['rec']: RadarPostureRule(*th, 0, k).classify(d['height'], d['offaxis']) for d in train}
        for hold in GRID_MIN_HOLD_S:
            rule = RadarPostureRule(*th, hold, k)
            hit = n = 0
            for d in train:
                radar, mocap = radar_events(d, rule, postures[d['rec']]), d['mocap'][mrule]
                hit += len(match_events(mocap, radar))
                n += len(mocap) + len(radar)
            f1 = 2 * hit / n if n else 0.0
            if best is None or f1 > best[1]:
                best = (rule, f1)
    return best


# ---- peak Doppler: which speed measure ----

def peak_pairs(recordings, rule_of, measure):
    """radar peak (measure, within 1 s) and MoCap peak at the counted MoCap transitions of the prepared recordings,
    under each recording's person's MoCap rule"""
    rows = []
    for d in recordings:
        m = d['mocap'][rule_of[d['person']]]
        m = m[m.kind.isin(COUNTED)]
        radar = add_peak(m[['t']].assign(peak_v=np.nan), d['t'], d['speeds'][measure]).peak_v.values
        rows.append(pd.DataFrame(dict(rec=d['rec'], person=d['person'], radar=radar, mocap=m.peak_v.values)))
    return pd.concat(rows, ignore_index=True).dropna()


def peak_measures(data, rule_of):
    """per candidate radar peak measure (v_peak and PEAK_FEATURES), at the counted MoCap transitions of all
    recordings: medians, bias, MAE, and the MAE without the bias (removed leaving the person out)"""
    rows = []
    for k in ('v_peak',) + PEAK_FEATURES:
        g = peak_pairs(data.values(), rule_of, k)
        diff = g.radar - g.mocap
        loso = g.person.map({p: diff[g.person != p].mean() for p in g.person.unique()})
        rows.append(dict(measure=k, transitions=len(g), radar_median=g.radar.median(), mocap_median=g.mocap.median(),
                         bias=diff.mean(), mae=diff.abs().mean(), mae_bias_removed=(diff - loso).abs().mean()))
    return pd.DataFrame(rows)


def calibrate_peak(data, rule_of, person):
    """(measure, training MAE): the speed quantile whose peaks are closest to MoCap's (mean absolute error) on every
    person but `person`"""
    train = others(data, person)
    mae = {k: (lambda g: (g.radar - g.mocap).abs().mean())(peak_pairs(train, rule_of, k)) for k in PEAK_FEATURES}
    best = min(mae, key=mae.get)
    return best, mae[best]


# ---- comparison: burst classifiers ----

def fit_burst_model(recordings, rule, features):
    """burst classifier trained on the in-window bursts of the prepared recordings, targets from MoCap under the
    MoCap posture rule; (model, bursts, transition bursts)"""
    b = pd.concat([d['bursts'][d['bursts'].in_window] for d in recordings], ignore_index=True)
    y = np.concatenate([burst_targets(d['bursts'][d['bursts'].in_window], d['mocap'][rule]) for d in recordings])
    return BurstModel(features).fit(b, y), len(b), int((y != NONE).sum())


def burst_classifier_fold(data, rule_of, scored, person, method, inner_folds=INNER_FOLDS):
    """one left-out person and classifier (CLASSIFIERS): detection threshold from out-of-fold classifications of the
    other persons, classifier trained on all of them, transitions of the person's recordings in `scored`.
    Returns (dict person, method, min_prob, train_f1; [(recording, radar transitions in its window)])."""
    features, rule = CLASSIFIERS[method], rule_of[person]
    rest = sorted({d['person'] for d in data.values()} - {person})
    train = others(data, person)
    oof = []
    for k in range(inner_folds):
        held = set(rest[k::inner_folds])
        m, _, _ = fit_burst_model([d for d in train if d['person'] not in held], rule, features)
        oof += [(m.classify(d['bursts']), d['mocap'][rule], window_filter(d)) for d in train if d['person'] in held]
    min_prob, train_f1 = calibrate_min_prob(oof)
    model, _, _ = fit_burst_model(train, rule, features)
    out = [(d['rec'], window_filter(d)(radar_transitions(model.classify(d['bursts']), min_prob)))
           for d in data.values() if d['person'] == person and d['rec'] in scored]
    return dict(person=person, method=method, min_prob=min_prob, train_f1=train_f1), out


# ---- comparison: hand rule ----

def hand_rule_baseline(data, rule_of, scored, grid=HAND_RULE_GRID):
    """per recording in `scored`: the hand rule (mobility.transitions.hand_rule_events) with the thresholds of `grid`
    that give the best pooled F1 (any kind) on the other persons, its transitions and matches against MoCap"""
    counts = {}
    for d in data.values():
        mocap = d['mocap'][rule_of[d['person']]]
        for g, params in enumerate(grid):
            r = in_window(hand_rule_events(d['bursts'], *params), d['t'], d['window'])
            counts[d['rec'], g] = (len(match_events(mocap, r, typed=False)), len(r), len(mocap))
    rows = []
    for person in sorted({d['person'] for d in data.values()}):
        f1 = [2 * sum(c[0] for (rec, gg), c in counts.items() if gg == g and data[rec]['person'] != person)
              / sum(c[1] + c[2] for (rec, gg), c in counts.items() if gg == g and data[rec]['person'] != person)
              for g in range(len(grid))]
        g = int(np.argmax(f1))
        for rec in (r for r in data if data[r]['person'] == person and r in scored):
            hit, n_r, n_m = counts[rec, g]
            rows.append(dict(rec=rec, v_min=grid[g][0], rise_min=grid[g][1], dur_min=grid[g][2],
                             mocap_events=n_m, radar_events=n_r, matched_any=hit,
                             f1_any=2 * hit / (n_m + n_r) if n_m + n_r else np.nan))
    return pd.DataFrame(rows)
