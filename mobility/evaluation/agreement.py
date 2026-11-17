"""
Per recording, all categories: the registered metrics of the radar and of MoCap on the same valid epochs, and how
well they agree across recordings.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..metrics import METRICS, compute_metrics
from ..metrics.category2 import set_bout_levels
from ..states import mocap_states, radar_states
from ..utils import person_of


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
    person = person_of(g.rec).values
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


def agreement_with_baseline(mt, bias_removed=False, ratio=False):
    """metric_agreement (same units on both sides), plus per metric: the medians of both sensors, the MAE relative to
    the mean MoCap value, and the MAE of a guess without any sensor: every recording gets the median MoCap value of
    the other persons. The script imposes much of what Categories 2-3 measure, so the radar is only informative where
    it beats this guess.
        bias_removed   also the MAE after removing the other persons' mean bias (a constant offset could be
                       calibrated away): mae_bias_removed
        ratio          also the radar MAE / the no-sensor MAE: vs_baseline"""
    agreement = metric_agreement(mt)
    person = person_of(mt.rec)
    for k, g in mt.groupby('metric', sort=False):
        i = agreement.metric == k
        p = person[g.index]
        agreement.loc[i, 'mocap_median'] = g.mocap.median()
        agreement.loc[i, 'radar_median'] = g.radar.median()
        agreement.loc[i, 'rel_mae'] = (g.radar - g.mocap).abs().mean() / g.mocap.abs().mean()
        if bias_removed:
            diff = g.radar - g.mocap
            agreement.loc[i, 'mae_bias_removed'] = np.nanmean(np.abs(diff - p.map(
                {q: diff[p != q].mean() for q in p.unique()})))
        base = np.array([g.mocap[p != q].median() for q in p])
        agreement.loc[i, 'baseline_mae'] = np.nanmean(np.abs(base - g.mocap.values))
    if ratio:
        agreement['vs_baseline'] = agreement.mean_abs_err / agreement.baseline_mae
    return agreement
