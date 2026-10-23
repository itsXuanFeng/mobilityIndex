"""
Radar vs MoCap: per epoch (AUC, and F1 with thresholds calibrated leave-one-subject-out) and per recording (the
registered metrics: error for times, rank correlation for energy).

Generalisation: when recording r (person p) is evaluated, nothing calibrated on p is used. MoCap active / inactive is
defined by gt_thresholds[p], the MoCap threshold calibrated against the activity labels without p, both for scoring r
and for training the radar threshold on the other persons' recordings.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

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


def epoch_pairs(session, radar_threshold, mocap_threshold, active_feature='n_cells', energy_feature='p_r4'):
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


def metric_table(sessions, thresholds, gt_thresholds, category=None, **radar_kw):
    """per recording and metric: radar value (with that recording's threshold) and MoCap value"""
    rows = []
    for s in sessions:
        rs = radar_states(s, thresholds[s.name], **radar_kw)
        ms = mocap_states(s, active_speed=gt_thresholds[s.person])
        rs.valid = ms.valid = rs.valid & ms.valid          # the same epochs on both sides
        rad, moc = compute_metrics(rs, category), compute_metrics(ms, category)
        for k in rad:
            rows.append(dict(rec=s.name, metric=k, unit=METRICS[k].unit, radar=rad[k], mocap=moc[k]))
    return pd.DataFrame(rows)


def metric_agreement(mt):
    """per metric across recordings. Same units on both sides (times): mean absolute error and bias (radar - MoCap).
    Different units (energy): only a rank correlation across recordings can check it."""
    rows = []
    for k, g in mt.groupby('metric', sort=False):
        row = dict(metric=k, n=len(g), unit=g.unit.iloc[0])
        if 'source units' in g.unit.iloc[0]:
            row.update(spearman=spearmanr(g.radar, g.mocap)[0])
        else:
            diff = g.radar - g.mocap
            row.update(mean_abs_err=diff.abs().mean(), bias=diff.mean())
        rows.append(row)
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
