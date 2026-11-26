"""
Category 2: bouts per epoch (inside a bout or not) and per bout (matched between radar and MoCap), and the radar bout
definition, calibrated leave-one-subject-out.

The radar definition has three calibrated numbers, each leaving the evaluated person out:
    distance slope   the radar level is corrected for the distance to the person (mobility.states.distance_corrected):
                     the slope of the radar level on log10 distance at a given MoCap level (loso_distance_slopes);
    entry            the active threshold on the corrected level, against MoCap active / inactive, as Category 1
                     calibrates its threshold (loso_corrected_entries);
    kappa            one margin on the corrected level, the same below and above the entry: exit = entry - kappa,
                     confirmation = entry + kappa, on the bout count against MoCap (loso_radar_bout_levels).
Both sensors' runs are also confirmed by their own posture transitions (Category 3; posture_events). Optionally, a
run must also reach a confirmation speed (m/s), calibrated together with kappa.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..bouts import BOUT_MATCH_IOU, level_bouts, match_bouts
from ..config import EPOCH_S, FPS
from ..labels import ACTIVE, INACTIVE
from ..metrics.category2 import in_bout, set_bout_levels
from ..posture import LYING, decode_postures, radar_height, radar_median_height, radar_moving, radar_offaxis
from ..transitions import in_window, mocap_transitions, radar_posture_transitions
from ..utils import best_threshold, runs
from .agreement import metric_rows

# radar margin on its log10 level: exit = entry - kappa, confirmation = entry + kappa; 0 = plain threshold
RADAR_KAPPAS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
RADAR_CONFIRM_SPEEDS = (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)      # speed confirmation candidates, m/s (option)
MIN_SEEN = 0.5          # distance slope: epochs where the radar sees the person in at least this share of the frames
BED_MARGIN_EPOCHS = 1   # a bout is in bed when the radar posture is lying in it or within this many epochs of it


# ---- posture transitions (Category 3) ----

def posture_events(session, radar_model, mocap_rule, threshold, feature='n_cells'):
    """the posture transitions inside the evaluation window, and the radar posture per frame: (radar events, MoCap
    events, radar posture). Radar: the posture model (mobility.posture.RadarPostureModel, calibrated without this
    person; movement gate = the Category 1 activity `feature` above its threshold), no further hold, as Category 3.
    MoCap: the MoCap posture rule (calibrated without this person)."""
    w = session.masks['window']
    inputs = (radar_height(session), radar_median_height(session), radar_offaxis(session),
              radar_moving(session, threshold, feature))
    posture = decode_postures(radar_model, [inputs])[0]
    radar = in_window(radar_posture_transitions(session.t, posture, session.radar['v_peak'], 1.0 / FPS), session.t, w)
    mocap = in_window(mocap_transitions(session, mocap_rule), session.t, w)
    return radar, mocap, posture


def lying_epochs(posture, n_epochs, n):
    """(E,) bool: the radar posture is lying in at least one frame of the epoch (n frames)"""
    p = np.asarray(posture)[:n_epochs * n].reshape(-1, n)
    return (p == LYING).any(1)


# ---- calibration, leaving the person out ----

def loso_distance_slopes(states, mocap_thr):
    """{person: slope b of the distance correction}, level + b * log10(distance / reference). For each person p, on
    the other persons' valid epochs where MoCap is active (p's fold threshold mocap_thr[p]) and the radar sees the
    person (extra['seen'] >= MIN_SEEN): least squares of the radar level on [log10 MoCap level, log_distance, 1];
    b = minus the coefficient of log_distance (the level falls with distance). states: [(person, radar States, MoCap
    States)], uncorrected."""
    out = {}
    for p in sorted({q for q, _, _ in states}):
        X, y = [], []
        for q, rs, ms in states:
            if q == p:
                continue
            ld = rs.extra['log_distance']
            ok = (rs.valid & (ms.level > mocap_thr[p]) & (rs.extra['seen'] >= MIN_SEEN) & ~np.isnan(ld)
                  & ~np.isnan(rs.level))
            X.append(np.c_[np.log10(ms.level[ok]), ld[ok], np.ones(ok.sum())])
            y.append(rs.level[ok])
        out[p] = -float(np.linalg.lstsq(np.concatenate(X), np.concatenate(y), rcond=None)[0][1])
    return out


def loso_corrected_entries(states, slopes, mocap_thr):
    """{person: entry threshold on the distance-corrected radar level}: for each person p, with p's slope, the
    threshold with the best balanced accuracy against MoCap active (p's fold threshold) on the other persons' valid
    epochs, as Category 1 calibrates its threshold (mobility.evaluation.loso_thresholds)"""
    out = {}
    for p in sorted({q for q, _, _ in states}):
        x, y = [], []
        for q, rs, ms in states:
            if q != p:
                level = rs.level + slopes[p] * np.nan_to_num(rs.extra['log_distance'])
                x.append(level[rs.valid])
                y.append(ms.level[rs.valid] > mocap_thr[p])
        out[p] = best_threshold(np.concatenate(x), np.concatenate(y))
    return out


def _mean_duration(b):
    return (b[:, 1] - b[:, 0]).mean() if len(b) else np.nan


def _training_sets(states, p, mocap_thr, mocap_factors, rules):
    """the recordings of all persons but p, with their MoCap bouts under p's fold: [(radar States, MoCap bouts)].
    The MoCap bouts are confirmed by MoCap's posture transitions when ms.extra['bout_transitions'] is set."""
    f_exit, f_conf = mocap_factors[p]
    return [(rs, level_bouts(ms.level, ms.valid, mocap_thr[p], f_exit * mocap_thr[p], f_conf * mocap_thr[p], rules[p],
                             transition=ms.extra.get('bout_transitions')))
            for q, rs, ms in states if q != p]


def _scores(train, entry, exit_threshold, confirm_threshold, rule, confirm_speed=None, slope=0.0):
    """training scores of one radar definition: summed absolute count error, mean absolute error of the mean bout
    duration, bout F1 (pooled). The radar level is corrected with `slope` (0: as measured)."""
    n_m = n_r = hit = count = 0
    dur = []
    for rs, mb in train:
        level = rs.level + slope * np.nan_to_num(rs.extra['log_distance']) if slope else rs.level
        rb = level_bouts(level, rs.valid, entry, exit_threshold, confirm_threshold, rule, rs.extra.get('speed'),
                         confirm_speed, rs.extra.get('bout_transitions'))
        n_m, n_r, hit = n_m + len(mb), n_r + len(rb), hit + len(match_bouts(mb, rb))
        count += abs(len(rb) - len(mb))
        dur.append(abs(_mean_duration(rb) - _mean_duration(mb)))
    return count, np.nanmean(dur), 2 * hit / (n_m + n_r) if n_m + n_r else np.nan


def loso_radar_bout_levels(states, radar_thr, mocap_thr, mocap_factors, rules, kappas=RADAR_KAPPAS, speeds=(None,),
                           slopes=None):
    """the radar margin kappa (exit = entry - kappa, confirmation = entry + kappa, on the log10 level) and, with
    speeds other than (None,), the confirmation speed, leaving the person out. For each person p, on the other persons'
    recordings: the (kappa, speed) whose bout count is closest to the MoCap bout count (sum of absolute differences over
    recordings). Ties: the smallest kappa, then no speed / the lowest speed (the least smoothing).
    Those recordings use p's fold: radar entry radar_thr[p] and, with slopes, the level corrected with slopes[p]; MoCap
    entry mocap_thr[p], exit and confirmation mocap_factors[p] x mocap_thr[p]; bout rule rules[p]; each sensor's own
    posture transitions where extra['bout_transitions'] is set (the transitions held out for that recording's person).
    states: [(person, radar States, MoCap States)], radar uncorrected; the speed option needs States.extra['speed'].
    Returns ({person: (kappa, speed or None)}, DataFrame of the training count error (per recording), mean-duration
    error and bout F1 per person, kappa and speed)."""
    out, rows = {}, []
    for p in sorted({q for q, _, _ in states}):
        train = _training_sets(states, p, mocap_thr, mocap_factors, rules)
        thr, slope = radar_thr[p], 0.0 if slopes is None else slopes[p]
        err = {}
        for k in kappas:
            for c in speeds:
                count, dur, f1 = _scores(train, thr, thr - k, thr + k, rules[p], c, slope)
                err[(k, c)] = count
                rows.append(dict(person=p, kappa=k, confirm_speed=np.nan if c is None else c,
                                 count_abs_err=count / len(train), duration_abs_err=dur, bout_f1=f1))
        out[p] = min(err, key=lambda q: (err[q], q[0], -1 if q[1] is None else q[1]))
    return out, pd.DataFrame(rows)


# ---- evaluation ----

def bout_frame(rs, ms):
    """per epoch of one recording (States with the bout rule and the bout levels set): the active calls (entry
    thresholds), whether the epoch is inside a radar bout and inside a MoCap bout, whether it contains a posture
    transition of each sensor (extra['bout_transitions']; False without) and whether the radar posture is lying
    (extra['lying']; False without)"""
    def extra(s, key):
        x = s.extra.get(key)
        return np.zeros(len(s.t), bool) if x is None else np.asarray(x, bool)[:len(s.t)]
    return pd.DataFrame(dict(t=rs.t, radar_active=rs.active, mocap_active=ms.active, radar_bout=in_bout(rs),
                             mocap_bout=in_bout(ms), evaluated=rs.valid & ms.valid,
                             radar_transition=extra(rs, 'bout_transitions'),
                             mocap_transition=extra(ms, 'bout_transitions'), lying=extra(rs, 'lying')))


def bout_events(pairs, epoch_s=EPOCH_S, min_iou=BOUT_MATCH_IOU, labels=None):
    """every bout of both sensors in one recording (pairs from bout_frame): start and end (s since the first epoch),
    duration, whether it is matched, for matched bouts the onset and offset error (radar - MoCap, s), whether it is
    in bed (the radar posture is lying in it or within BED_MARGIN_EPOCHS of it: the same posture for both sensors, as
    it would be in deployment), and with labels (one per epoch) the label at the middle of the bout"""
    t0 = pairs.t.values[0]
    rb, mb = runs(pairs.radar_bout.values), runs(pairs.mocap_bout.values)
    m = match_bouts(mb, rb, min_iou)
    lying = pairs.lying.values
    rows = []
    for src, b, partner in (('mocap', mb, m), ('radar', rb, {j: i for i, j in m.items()})):
        for i, (s, e) in enumerate(b):
            row = dict(source=src, start_s=pairs.t.values[s] - t0, end_s=pairs.t.values[e - 1] - t0 + epoch_s,
                       duration_s=(e - s) * epoch_s, matched=i in partner,
                       in_bed=bool(lying[max(s - BED_MARGIN_EPOCHS, 0):e + BED_MARGIN_EPOCHS].any()),
                       label=labels[(s + e - 1) // 2] if labels is not None else '')
            if i in partner and src == 'radar':
                ms, me = mb[partner[i]]
                row.update(onset_err_s=(s - ms) * epoch_s, offset_err_s=(e - me) * epoch_s)
            rows.append(row)
    return pd.DataFrame(rows, columns=['source', 'start_s', 'end_s', 'duration_s', 'matched', 'onset_err_s',
                                       'offset_err_s', 'in_bed', 'label'])


def split_metric_rows(events):
    """per recording, the number of bouts and the mean bout duration in bed and out of bed (bout_events' in_bed), in
    the form of metric_rows (for agreement_with_baseline): events of all recordings, with rec"""
    rows = []
    for rec, g in events.groupby('rec', sort=False):
        for part, sel in (('in_bed', g.in_bed), ('out_of_bed', ~g.in_bed)):
            per = {src: g[sel & (g.source == src)].duration_s for src in ('radar', 'mocap')}
            rows += [dict(rec=rec, metric=f'n_bouts_{part}', unit='count', radar=float(len(per['radar'])),
                          mocap=float(len(per['mocap']))),
                     dict(rec=rec, metric=f'bout_duration_mean_{part}', unit='s', radar=per['radar'].mean(),
                          mocap=per['mocap'].mean())]
    return pd.DataFrame(rows)


def _f1_labels(bout, cls, evaluated):
    """F1 of in-bout calls against the activity labels (active / inactive label epochs); NaN without such epochs"""
    ok = evaluated & ((cls == ACTIVE) | (cls == INACTIVE))
    return f1_score(cls[ok] == ACTIVE, bout[ok]) if ok.any() else np.nan


def bout_scores(pairs, events, classes=None):
    """one recording, MoCap as the reference. Per evaluated epoch: F1 of the radar's active calls and of its in-bout
    calls; epochs with a posture transition of each sensor, and how many of them are inside that sensor's bouts (a
    consistency check between Categories 2 and 3). Per bout: precision, recall, F1 of the matching, and the onset /
    offset error of the matched bouts. With
    classes (label class per epoch, mobility.labels.time_classes): F1 of each sensor's in-bout calls against the
    labels, a reference independent of MoCap."""
    e = pairs[pairs.evaluated]
    n_m, n_r = (events.source == 'mocap').sum(), (events.source == 'radar').sum()
    hit = int(events[events.source == 'radar'].matched.sum())
    p, r = hit / n_r if n_r else np.nan, hit / n_m if n_m else np.nan
    matched = events.dropna(subset=['onset_err_s'])
    out = dict(epochs=len(e), f1_active=f1_score(e.mocap_active, e.radar_active),
               f1_bout_epochs=f1_score(e.mocap_bout, e.radar_bout), mocap_bouts=n_m, radar_bouts=n_r, matched=hit,
               bout_precision=p, bout_recall=r, bout_f1=2 * hit / (n_m + n_r) if n_m + n_r else np.nan,
               onset_mae_s=matched.onset_err_s.abs().mean(), offset_mae_s=matched.offset_err_s.abs().mean(),
               radar_transitions=int(e.radar_transition.sum()), mocap_transitions=int(e.mocap_transition.sum()),
               radar_transitions_in_bout=int((e.radar_transition & e.radar_bout).sum()),
               mocap_transitions_in_bout=int((e.mocap_transition & e.mocap_bout).sum()))
    if classes is not None:
        cls, ev = np.asarray(classes, object), pairs.evaluated.values
        out.update(f1_labels_radar=_f1_labels(pairs.radar_bout.values, cls, ev),
                   f1_labels_mocap=_f1_labels(pairs.mocap_bout.values, cls, ev))
    return out


def evaluate_bouts(rec, rs, ms, labels, rule, levels, classes=None):
    """bouts, scores and metrics of one recording with the given bout rule and levels: ((radar exit, radar
    confirmation), (MoCap exit, MoCap confirmation)[, radar confirmation speed]) thresholds. labels: the activity label
    per epoch (bouts.csv); classes: the label class per epoch (F1 against the labels). Returns (bout_events with rec,
    dict of bout_scores with rec, DataFrame of the Category 2 metric rows)"""
    rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
    set_bout_levels(rs, ms, *levels)
    frame = bout_frame(rs, ms)
    events = bout_events(frame, labels=labels).assign(rec=rec)
    return events, dict(rec=rec, **bout_scores(frame, events, classes)), pd.DataFrame(metric_rows(rec, rs, ms, 2))
