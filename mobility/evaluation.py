"""
Radar vs MoCap: per epoch (AUC, and F1 with thresholds calibrated leave-one-subject-out) and per recording (the
registered metrics: error for times; for energy, rank correlation and the error after a conversion to MoCap units
fitted leaving the person out). Category 2 adds the bouts: per epoch (inside a bout
or not) and per bout (matched between radar and MoCap). Category 3 adds the postural transitions, per event.

Generalisation: when recording r (person p) is evaluated, nothing calibrated on p is used. MoCap active / inactive is
defined by gt_thresholds[p], the MoCap threshold calibrated against the activity labels without p, both for scoring r
and for training the radar threshold on the other persons' recordings.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from .bouts import hysteresis, in_bouts, runs
from .config import EPOCH_S, FPS
from .labels import label_class
from .metrics import METRICS, compute_metrics
from .states import feature_level, mocap_states, radar_states
from .utils import best_threshold, epoch_mean


def epoch_table(sessions, features, gt_thresholds):
    """one row per epoch with MoCap data: recording, person, label, MoCap energy / speed, active (by the person's
    own fold threshold), distance to the radar, window flags, and the epoch level of each radar feature"""
    n = int(round(EPOCH_S * FPS))
    out = []
    for s in sessions:
        ms = mocap_states(s, active_speed=gt_thresholds[s.person], window='has_mocap')
        E = len(ms.t)
        df = pd.DataFrame(dict(rec=s.name, person=s.person, t=ms.t, label=s.label_per_frame()[n // 2::n][:E],
                               ke=ms.energy, speed=ms.level, active=ms.active,
                               dist=epoch_mean(s.mocap['dist_cog'], n)[:E]))
        for m in ('window', 'covered', 'in_script', 'no_caregiver'):
            df[m] = epoch_mean(s.masks[m], n)[:E] >= 0.5
        for f in features:
            df[f] = feature_level(s, f, n)[:E]
        out.append(df[ms.valid])
    return pd.concat(out, ignore_index=True)


def epoch_pairs(session, radar_threshold, mocap_threshold, active_feature='n_cells', energy_feature='e_move'):
    """per 1 s epoch over the whole recording, the Category 1 states side by side: movement energy and active call
    of the radar and of MoCap, whether the epoch is evaluated, its label and label class, distance to the radar"""
    n = int(round(EPOCH_S * FPS))
    rs = radar_states(session, radar_threshold, active_feature, energy_feature)
    ms = mocap_states(session, mocap_threshold)
    E = len(rs.t)
    label = session.label_per_frame()[n // 2::n][:E]
    return pd.DataFrame(dict(t=rs.t, radar_energy=rs.energy, mocap_energy=ms.energy, radar_active=rs.active,
                             mocap_active=ms.active, evaluated=rs.valid & ms.valid, label=label,
                             label_class=[label_class(x) if x else None for x in label],
                             dist=epoch_mean(session.mocap['dist_cog'], n)[:E]))


def loso_thresholds(table, feature, gt_thresholds):
    """{recording: radar threshold calibrated on the recordings of all other persons}, with MoCap active defined by
    the evaluated person's fold threshold"""
    w = table[table.window]
    out = {}
    for r, person in w[['rec', 'person']].drop_duplicates().itertuples(index=False):
        train = w[w.person != person]
        out[r] = best_threshold(train[feature].values, (train.speed > gt_thresholds[person]).values)
    return out


def epoch_scores(table, feature, thresholds):
    """per recording, MoCap active as the positive class: AUC of the feature level (threshold-free) and F1 of the
    active calls with the recording's threshold"""
    rows = []
    for r, g in table[table.window].groupby('rec', sort=False):
        g = g[~g[feature].isna()]
        pred = g[feature].values > thresholds[r]
        rows.append(dict(feature=feature, rec=r, epochs=len(g), auc=roc_auc_score(g.active, g[feature]),
                         f1=f1_score(g.active, pred)))
    return pd.DataFrame(rows)


def paired_states(session, radar_threshold, mocap_threshold, **radar_kw):
    """radar and MoCap States of one recording, valid on the same epochs"""
    rs = radar_states(session, radar_threshold, **radar_kw)
    ms = mocap_states(session, active_speed=mocap_threshold)
    rs.valid = ms.valid = rs.valid & ms.valid
    return rs, ms


def metric_rows(name, rs, ms, category=None):
    """per metric: radar and MoCap value of one recording"""
    rad, moc = compute_metrics(rs, category), compute_metrics(ms, category)
    return [dict(rec=name, metric=k, unit=METRICS[k].unit, radar=rad[k], mocap=moc[k]) for k in rad]


def metric_table(sessions, thresholds, gt_thresholds, category=None, bout_rules=None, bout_levels=None,
                 transitions=None, postures=None, **radar_kw):
    """per recording and metric: radar value (with that recording's threshold) and MoCap value. Category 2 needs
    bout_rules: {person: BoutRule calibrated without that person} and bout_levels: {recording: ((radar exit, radar
    confirmation), (MoCap exit, MoCap confirmation)) thresholds} (see set_bout_levels). Category 3 needs transitions:
    {recording: (radar events, MoCap events)} (mobility.transitions) and postures: {recording: (radar, MoCap) posture
    per epoch}."""
    rows = []
    for s in sessions:
        rs, ms = paired_states(s, thresholds[s.name], gt_thresholds[s.person], **radar_kw)
        if bout_rules is not None:
            rs.extra['bout_rule'] = ms.extra['bout_rule'] = bout_rules[s.person]
        if bout_levels is not None:
            set_bout_levels(rs, ms, *bout_levels[s.name])
        if transitions is not None:
            rs.transitions, ms.transitions = transitions[s.name]
        if postures is not None:
            rs.posture, ms.posture = (p[:len(rs.t)] for p in postures[s.name])
        rows += metric_rows(s.name, rs, ms, category)
    return pd.DataFrame(rows)


def energy_conversion(g):
    """for a metric in source units (energy), per recording: the radar value in MoCap units, with one factor
    (sum of MoCap / sum of radar) fitted on the other persons' recordings; and the no-radar baseline, the other
    persons' mean MoCap value. g: the metric's rows of metric_table"""
    person = g.rec.str.split('/').str[0].values
    radar, mocap = g.radar.values, g.mocap.values
    pred = np.array([radar[i] * mocap[person != p].sum() / radar[person != p].sum() for i, p in enumerate(person)])
    base = np.array([mocap[person != p].mean() for p in person])
    return pd.DataFrame(dict(rec=g.rec.values, person=person, mocap=mocap, converted=pred, baseline=base))


def metric_agreement(mt):
    """per metric across recordings. Same units on both sides (times): mean absolute error and bias (radar - MoCap).
    Different units (energy): Spearman across recordings, and the mean absolute error and bias after the conversion
    to MoCap units (energy_conversion), in % of the mean MoCap value, next to the no-radar baseline's error."""
    rows = []
    for k, g in mt.groupby('metric', sort=False):
        row = dict(metric=k, n=len(g), unit=g.unit.iloc[0])
        if 'source units' in g.unit.iloc[0]:
            c = energy_conversion(g)
            m = c.mocap.mean()
            row.update(spearman=spearmanr(g.radar, g.mocap)[0],
                       error_pct=100 * (c.converted - c.mocap).abs().mean() / m,
                       bias_pct=100 * (c.converted - c.mocap).mean() / m,
                       baseline_error_pct=100 * (c.baseline - c.mocap).abs().mean() / m)
        else:
            diff = g.radar - g.mocap
            row.update(mean_abs_err=diff.abs().mean(), bias=diff.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def per_person(table, feature, thresholds, mt):
    """per person, over their evaluated epochs pooled: AUC of the feature, F1 of the active and of the inactive
    calls (each recording with its own threshold); and the energy error after the conversion to MoCap units, in %
    of the person's mean MoCap energy"""
    w = table[table.window & table[feature].notna()]
    pred = w[feature].values > w.rec.map(thresholds).values
    c = energy_conversion(mt[mt.metric == 'total_activity_energy'])
    rows = []
    for p, g in w.groupby('person', sort=False):
        y, yhat = g.active.values, pred[(w.person == p).values]
        e = c[c.person == p]
        rows.append(dict(person=p, recordings=g.rec.nunique(), epochs=len(g), auc=roc_auc_score(y, g[feature]),
                         f1_active=f1_score(y, yhat), f1_inactive=f1_score(~y, ~yhat),
                         energy_error_pct=100 * (e.converted - e.mocap).abs().mean() / e.mocap.mean(),
                         energy_bias_pct=100 * (e.converted - e.mocap).mean() / e.mocap.mean()))
    return pd.DataFrame(rows)


def per_label(table, feature, threshold):
    """per scripted activity: share of epochs active by MoCap and by the radar, coverage, distance"""
    t = table[table.label != ''].assign(radar_active=lambda d: d[feature] > threshold)
    g = t.groupby('label', sort=False).agg(epochs=('ke', 'size'), covered=('covered', 'mean'),
                                           alone=('no_caregiver', 'mean'), mocap_active=('active', 'mean'),
                                           radar_active=('radar_active', 'mean'), dist_m=('dist', 'median'))
    g['radar_minus_mocap'] = g.radar_active - g.mocap_active
    return g


def distance_residuals(table, feature, bins=(0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)):
    """active epochs in the window: residual of the feature level after a linear fit on log10 MoCap energy, per
    distance radar - CoG. Flat = the feature doesn't depend on where the person is."""
    a = table[table.window & table.active & ~table[feature].isna()]
    A = np.c_[np.log10(a.ke), np.ones(len(a))]
    res = a[feature].values - A @ np.linalg.lstsq(A, a[feature].values, rcond=None)[0]
    return pd.Series(res).groupby(pd.cut(a.dist.values, bins)).agg(['size', 'median'])


# ---- Category 2: bouts ----

BOUT_MATCH_IOU = 0.5   # a radar bout and a MoCap bout match when they overlap by at least this intersection over union
RADAR_EXIT_DELTAS = (0, 0.1, 0.2, 0.3, 0.4)                       # radar exit = entry - delta, on its log10 level
RADAR_CONFIRM_EPS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)   # radar confirmation = entry + eps


def set_bout_levels(rs, ms, radar, mocap):
    """the Category 2 level thresholds of each sensor: radar and mocap are (exit, confirmation) thresholds"""
    rs.extra['exit_threshold'], rs.extra['confirm_threshold'] = radar
    ms.extra['exit_threshold'], ms.extra['confirm_threshold'] = mocap


def bout_frame(rs, ms):
    """per epoch of one recording (States with the bout rule and the bout levels set): the Category 1 active calls and
    whether the epoch is inside a radar bout and inside a MoCap bout"""
    from .metrics.category2 import in_bout
    return pd.DataFrame(dict(t=rs.t, radar_active=rs.active, mocap_active=ms.active, radar_bout=in_bout(rs),
                             mocap_bout=in_bout(ms), evaluated=rs.valid & ms.valid))


def _bouts(level, valid, entry, exit_threshold, confirm_threshold, rule):
    return runs(in_bouts(hysteresis(level > entry, level, valid, exit_threshold, confirm_threshold), valid, rule))


def loso_radar_bout_levels(states, radar_thr, mocap_thr, mocap_factors, rules, deltas=RADAR_EXIT_DELTAS,
                           eps=RADAR_CONFIRM_EPS):
    """radar exit and confirmation thresholds (entry - delta, entry + eps on the log10 level), leaving the person
    out: for each person p, the pair whose bout count is closest to the MoCap bout count (sum of absolute differences)
    on the other persons' recordings. As for the active thresholds, those recordings use p's fold: radar entry
    radar_thr[p]; MoCap entry mocap_thr[p], exit and confirmation mocap_factors[p] x mocap_thr[p]; bout rule rules[p].
    states: [(person, radar States, MoCap States)]. Ties: the smallest eps, then the smallest delta.
    Returns ({person: (delta, eps)}, DataFrame of the training count error and bout F1 per person and pair)."""
    out, rows = {}, []
    for p in sorted({q for q, _, _ in states}):
        f_exit, f_conf = mocap_factors[p]
        train = [(rs, ms, _bouts(ms.level, ms.valid, mocap_thr[p], f_exit * mocap_thr[p], f_conf * mocap_thr[p],
                                 rules[p])) for q, rs, ms in states if q != p]
        err = {}
        for e in eps:
            for d in deltas:
                n_m = n_r = hit = abs_err = 0
                for rs, ms, mb in train:
                    rb = _bouts(rs.level, rs.valid, radar_thr[p], radar_thr[p] - d, radar_thr[p] + e, rules[p])
                    n_m, n_r, hit = n_m + len(mb), n_r + len(rb), hit + len(match_bouts(mb, rb))
                    abs_err += abs(len(rb) - len(mb))
                err[(d, e)] = abs_err
                rows.append(dict(person=p, delta=d, eps=e, count_abs_err=abs_err / len(train),
                                 bout_f1=2 * hit / (n_m + n_r)))
        out[p] = min(err, key=lambda k: (err[k], k[1], k[0]))
    return out, pd.DataFrame(rows)


def match_bouts(ref, est, min_iou=BOUT_MATCH_IOU):
    """one-to-one matching of (n, 2) and (m, 2) [start, end) bouts, greedily by intersection over union.
    Returns {index in ref: index in est}."""
    if not len(ref) or not len(est):
        return {}
    inter = np.clip(np.minimum(ref[:, None, 1], est[None, :, 1]) - np.maximum(ref[:, None, 0], est[None, :, 0]), 0,
                    None)
    union = (ref[:, None, 1] - ref[:, None, 0]) + (est[None, :, 1] - est[None, :, 0]) - inter
    iou = inter / union
    out, used = {}, set()
    for k in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(k, iou.shape)
        if iou[i, j] < min_iou:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def bout_events(pairs, epoch_s=EPOCH_S, min_iou=BOUT_MATCH_IOU, labels=None):
    """every bout of both sensors in one recording (pairs from bout_frame): start and end (s since the first epoch),
    duration, whether it is matched, for matched bouts the onset and offset error (radar - MoCap, s), and with labels
    (one per epoch) the label at the middle of the bout"""
    t0 = pairs.t.values[0]
    rb, mb = runs(pairs.radar_bout.values), runs(pairs.mocap_bout.values)
    m = match_bouts(mb, rb, min_iou)
    rows = []
    for src, b, partner in (('mocap', mb, m), ('radar', rb, {j: i for i, j in m.items()})):
        for i, (s, e) in enumerate(b):
            row = dict(source=src, start_s=pairs.t.values[s] - t0, end_s=pairs.t.values[e - 1] - t0 + epoch_s,
                       duration_s=(e - s) * epoch_s, matched=i in partner,
                       label=labels[(s + e - 1) // 2] if labels is not None else '')
            if i in partner and src == 'radar':
                ms, me = mb[partner[i]]
                row.update(onset_err_s=(s - ms) * epoch_s, offset_err_s=(e - me) * epoch_s)
            rows.append(row)
    return pd.DataFrame(rows, columns=['source', 'start_s', 'end_s', 'duration_s', 'matched', 'onset_err_s',
                                       'offset_err_s', 'label'])


def bout_scores(pairs, events):
    """one recording, MoCap as the reference. Per evaluated epoch: F1 of the radar's active calls and of its in-bout
    calls. Per bout: precision, recall, F1 of the matching, and the onset / offset error of the matched bouts."""
    e = pairs[pairs.evaluated]
    n_m, n_r = (events.source == 'mocap').sum(), (events.source == 'radar').sum()
    hit = int(events[events.source == 'radar'].matched.sum())
    p, r = hit / n_r if n_r else np.nan, hit / n_m if n_m else np.nan
    matched = events.dropna(subset=['onset_err_s'])
    return dict(epochs=len(e), f1_active=f1_score(e.mocap_active, e.radar_active),
                f1_bout_epochs=f1_score(e.mocap_bout, e.radar_bout), mocap_bouts=n_m, radar_bouts=n_r, matched=hit,
                bout_precision=p, bout_recall=r, bout_f1=2 * hit / (n_m + n_r) if n_m + n_r else np.nan,
                onset_mae_s=matched.onset_err_s.abs().mean(), offset_mae_s=matched.offset_err_s.abs().mean())


# ---- Category 3: postural transitions ----

def transition_scores(radar, mocap, tol=None):
    """one recording, MoCap as the reference, events inside the evaluation window. Typed: matched within tol and of
    the same kind. Any: matched within tol whatever the kind (a transition is detected, its kind may be wrong).
    Matched pairs (typed): time error and peak-speed error (radar - MoCap)."""
    from .transitions import MATCH_TOL_S, match_events
    tol = MATCH_TOL_S if tol is None else tol
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


def transition_events(radar, mocap, t0, tol=None):
    """every event of both sensors in one recording: time (s since t0), kind, direction, peak speed; whether it is
    matched (typed), and the kind of the other sensor's event matched in time whatever the kind ('' if none); for
    MoCap events, the distance from the radar axis (offaxis_m)"""
    from .transitions import MATCH_TOL_S, match_events
    tol = MATCH_TOL_S if tol is None else tol
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


def posture_scores(radar, mocap, height, valid):
    """one recording, per 1 s epoch, MoCap posture as the reference (epochs valid and with a known MoCap posture):
    F1 of the radar posture for each posture MoCap shows, their mean, and the AUC of the radar height for each pair
    of neighbouring postures (threshold-free: how well height separates them); seen = share of the epochs where the
    radar sees the person"""
    from .posture import FLOOR, LYING, POSTURES, SITTING, STANDING, UNKNOWN
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
