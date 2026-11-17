"""
Category 1 per epoch: the radar activity level and active calls against MoCap active, with the radar threshold
calibrated leave-one-subject-out; per person, per scripted activity and per distance.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from ..config import EPOCH_FRAMES, EPOCH_S
from ..labels import label_class
from ..states import feature_level, mocap_states, radar_states
from ..utils import best_threshold, epoch_mean, log_level
from .agreement import energy_conversion


def epoch_table(sessions, features, gt_thresholds):
    """one row per epoch with MoCap data: recording, person, label, MoCap energy / speed, active (by the person's
    own fold threshold), distance to the radar, window flags, and the epoch level of each radar feature"""
    n = EPOCH_FRAMES
    out = []
    for s in sessions:
        ms = mocap_states(s, active_speed=gt_thresholds[s.person], window='has_mocap')
        E = len(ms.t)
        df = pd.DataFrame(dict(rec=s.name, person=s.person, t=ms.t, label=s.label_per_epoch(n)[:E],
                               ke=ms.energy, speed=ms.level, active=ms.active,
                               dist=epoch_mean(s.mocap['dist_cog'], n)[:E]))
        for m in ('window', 'covered', 'in_script', 'no_caregiver'):
            df[m] = s.epoch_mask(m, n)[:E]
        for f in features:
            df[f] = feature_level(s, f, n)[:E]
        out.append(df[ms.valid])
    return pd.concat(out, ignore_index=True)


def count_epoch_table(session, counts, active_speed, with_label=False):
    """the epoch_table rows of one recording for per-frame counts of detected cells that are not registered features
    (the detector explorations): MoCap speed and active, the window flag, and the epoch level of each count
    {name: (T,)} and of n_cells (the current detector), as feature_level for a log feature. with_label: also the
    label and the time since the first epoch (t_s)"""
    n = EPOCH_FRAMES
    s = session
    ms = mocap_states(s, active_speed=active_speed, window='has_mocap')
    E = len(ms.t)
    cols = dict(rec=s.name, person=s.person, speed=ms.level, active=ms.active, window=s.epoch_mask('window', n)[:E])
    if with_label:
        cols.update(label=s.label_per_epoch(n)[:E], t_s=np.arange(E) * EPOCH_S)
    df = pd.DataFrame(cols)
    for name, x in counts.items():
        df[name] = epoch_mean(log_level(x), n)[:E]
    df['n_cells'] = epoch_mean(log_level(s.radar['n_cells']), n)[:E]
    return df[ms.valid]


def epoch_pairs(session, radar_threshold, mocap_threshold, active_feature='n_cells', energy_feature='e_move'):
    """per 1 s epoch over the whole recording, the Category 1 states side by side: movement energy and active call
    of the radar and of MoCap, whether the epoch is evaluated, its label and label class, distance to the radar"""
    n = EPOCH_FRAMES
    rs = radar_states(session, radar_threshold, active_feature, energy_feature)
    ms = mocap_states(session, mocap_threshold)
    E = len(rs.t)
    label = session.label_per_epoch(n)[:E]
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
