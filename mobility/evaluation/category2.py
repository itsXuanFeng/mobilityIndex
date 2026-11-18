"""
Category 2: bouts per epoch (inside a bout or not) and per bout (matched between radar and MoCap), and the radar
exit and confirmation thresholds, calibrated leave-one-subject-out against the MoCap bouts: the confirmation on the
bout count (it decides which bouts exist), the exit on the mean bout duration (it decides where they end).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..bouts import BOUT_MATCH_IOU, level_bouts, match_bouts
from ..config import EPOCH_S
from ..metrics.category2 import in_bout, set_bout_levels
from ..utils import runs
from .agreement import metric_rows

RADAR_EXIT_DELTAS = (0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2)   # radar exit = entry - delta, on its log10 level
RADAR_CONFIRM_EPS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)   # radar confirmation = entry + eps


def bout_frame(rs, ms):
    """per epoch of one recording (States with the bout rule and the bout levels set): the Category 1 active calls and
    whether the epoch is inside a radar bout and inside a MoCap bout"""
    return pd.DataFrame(dict(t=rs.t, radar_active=rs.active, mocap_active=ms.active, radar_bout=in_bout(rs),
                             mocap_bout=in_bout(ms), evaluated=rs.valid & ms.valid))


def loso_radar_bout_levels(states, radar_thr, mocap_thr, mocap_factors, rules, deltas=RADAR_EXIT_DELTAS,
                           eps=RADAR_CONFIRM_EPS):
    """radar exit and confirmation thresholds (entry - delta, entry + eps on the log10 level), leaving the person
    out. For each person p, on the other persons' recordings:
        1. eps: the confirmation of the pair (delta, eps) whose bout count is closest to the MoCap bout count (sum of
           absolute differences over recordings). The count hardly depends on delta;
        2. delta: at that eps, the exit whose mean bout duration is closest to the MoCap mean bout duration (mean
           absolute difference over recordings).
    As for the active thresholds, those recordings use p's fold: radar entry radar_thr[p]; MoCap entry mocap_thr[p],
    exit and confirmation mocap_factors[p] x mocap_thr[p]; bout rule rules[p]. states: [(person, radar States, MoCap
    States)]. Ties: the smallest eps, then the smallest delta.
    Returns ({person: (delta, eps)}, DataFrame of the training count error, mean-duration error and bout F1 per person
    and pair)."""
    def mean_duration(b):
        return (b[:, 1] - b[:, 0]).mean() if len(b) else np.nan

    out, rows = {}, []
    for p in sorted({q for q, _, _ in states}):
        f_exit, f_conf = mocap_factors[p]
        train = [(rs, ms, level_bouts(ms.level, ms.valid, mocap_thr[p], f_exit * mocap_thr[p], f_conf * mocap_thr[p],
                                      rules[p])) for q, rs, ms in states if q != p]
        count_err, dur_err = {}, {}
        for e in eps:
            for d in deltas:
                n_m = n_r = hit = abs_err = 0
                dur = []
                for rs, ms, mb in train:
                    rb = level_bouts(rs.level, rs.valid, radar_thr[p], radar_thr[p] - d, radar_thr[p] + e, rules[p])
                    n_m, n_r, hit = n_m + len(mb), n_r + len(rb), hit + len(match_bouts(mb, rb))
                    abs_err += abs(len(rb) - len(mb))
                    dur.append(abs(mean_duration(rb) - mean_duration(mb)))
                count_err[(d, e)], dur_err[(d, e)] = abs_err, np.nanmean(dur)
                rows.append(dict(person=p, delta=d, eps=e, count_abs_err=abs_err / len(train),
                                 duration_abs_err=dur_err[(d, e)], bout_f1=2 * hit / (n_m + n_r)))
        e_best = min(count_err, key=lambda k: (count_err[k], k[1], k[0]))[1]
        d_best = min(deltas, key=lambda d: (round(dur_err[(d, e_best)], 9), d))
        out[p] = (d_best, e_best)
    return out, pd.DataFrame(rows)


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


def evaluate_bouts(rec, rs, ms, labels, rule, levels):
    """bouts, scores and metrics of one recording with the given bout rule and levels: ((radar exit, radar
    confirmation), (MoCap exit, MoCap confirmation)) thresholds. Returns (bout_events with rec, dict of bout_scores
    with rec, DataFrame of the Category 2 metric rows)"""
    rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
    set_bout_levels(rs, ms, *levels)
    frame = bout_frame(rs, ms)
    events = bout_events(frame, labels=labels).assign(rec=rec)
    return events, dict(rec=rec, **bout_scores(frame, events)), pd.DataFrame(metric_rows(rec, rs, ms, 2))
