"""
Category 3 (postural transitions, time in each posture): radar vs MoCap on the recordings evaluated in Category 1.
Run first:
    scripts/evaluate_category1.py     (thresholds.csv: the recordings, the evaluation window)
    scripts/calibrate_posture.py      (MoCap posture rule per left-out person)
    scripts/angle_pass.py             (points.npz: the angle; the person's height per frame is derived from it)

    python scripts/evaluate_category3.py --workers 16
    python scripts/evaluate_category3.py --only person_1/scenario_1 person_5/scenario_3   (calibrates on all)

Main method: the radar posture from the height of the top of the body (mobility.posture.RadarPostureRule), and the
transitions where it changes and holds, the same rule as for MoCap. For each person, on the *other* persons, with MoCap
defined by that person's MoCap posture rule: for each off-axis correction offaxis_k, the three height thresholds
maximise the balanced accuracy of their split per 1 s epoch against the MoCap posture; then offaxis_k and min_hold_s
together maximise the transition F1 against MoCap.

Peak Doppler: the radar speed of the person's moving points, at a power-weighted quantile (mobility.radar.angle,
PEAK_QUANTILES) chosen for each person on the *other* persons: the one whose peak within 1 s of their MoCap
transitions is closest to the MoCap peak (mean absolute error). The fastest strong cell (feature v_peak, the first
version) is reported next to it.

Comparisons, also leave-one-subject-out:
    classifier          burst classifier on range and Doppler only (the first version, docs/category3.md)
    classifier+angle    the same with the angle burst features (height before / after, distance from the axis)
    hand rule           thresholds on burst speed, range change, duration (finds transitions, not their kind)

Writes to OUTPUT_ROOT/category3/:
    radar_posture_rule.csv  per left-out person (and ALL): height thresholds, hold, training F1
    peak_rule.csv           per left-out person (and ALL): the speed quantile used for the peak, its training MAE
    peak_measures.csv       per candidate peak measure, on the MoCap transitions: median, bias, MAE, MAE without the
                            bias (removed leaving the person out)
    posture_scores.csv      per recording: F1 per posture, AUC of the height per pair of neighbouring postures, seen
    posture_confusion.csv   epochs, MoCap posture x radar posture, pooled
    event_scores.csv        per recording, main method: transitions, matched, precision, recall, F1, time, peak error
    events.csv              every transition of both sensors, main method
    per_kind.csv            per kind, pooled, main method
    offaxis.csv             counted MoCap transitions found by the radar, by distance from the radar axis
    methods.csv             per recording and method: transitions, matched, F1 (same kind and any kind)
    thresholds.csv          per left-out person and classifier: detection threshold, out-of-fold F1
    metrics.csv             per recording: the Category 3 metrics, radar (main method) vs MoCap
    agreement.csv           per metric: MAE, bias, medians, relative MAE, MAE after removing the bias of the other
                            persons, no-sensor baseline MAE
    rule_baseline.csv       per recording: the hand rule
"""
import argparse
import itertools
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '6')    # threads per classifier; folds run in parallel (--fold-workers)
import traceback  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.config import EPOCH_S, FPS, OUTPUT_ROOT  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.radar.angle import PEAK_FEATURES  # noqa: E402
from mobility.posture import (GRID_MIN_HOLD_S, GRID_OFFAXIS_K, POSTURES, RADAR_POSTURE_RULE_FILE,  # noqa: E402
                              UNKNOWN, RadarPostureRule, calibrate_radar_thresholds, epoch_posture, posture_rules,
                              radar_height, radar_offaxis)
from mobility.session import load_session  # noqa: E402
from mobility.transitions import (ANGLE_BURST_FEATURES, BURST_FEATURES, COUNTED, NMS_S, BurstModel,  # noqa: E402
                                  add_peak, burst_targets, bursts, calibrate_min_prob, in_window, match_events,
                                  mocap_transitions, radar_posture_transitions, radar_transitions)
from mobility.utils import epoch_mean  # noqa: E402

N = int(round(EPOCH_S * FPS))   # frames per epoch
INNER_FOLDS = 5    # classifier detection threshold: out-of-fold classifications of the training persons
CLASSIFIERS = {'classifier': BURST_FEATURES, 'classifier+angle': ANGLE_BURST_FEATURES}
# hand-rule baseline: a burst is a transition when fast, long and its range change agrees with its Doppler sign
RULE_GRID = list(itertools.product((0.1, 0.15, 0.2, 0.25, 0.3), (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                                   (0.3, 0.5, 0.8, 1.0)))     # v_smooth (m/s), rise (m), duration (s)


def prepare(job):
    """one recording, in a worker: radar height, bursts, and the MoCap posture and transitions under every MoCap
    posture rule a fold may use"""
    rec, rules = job
    try:
        s = load_session(*parse_recording(rec))
        w = s.masks['window']
        b = bursts(s)
        b['in_window'] = w[np.clip(np.searchsorted(s.t, b.t.values), 0, len(s.t) - 1)]
        h, off = radar_height(s), radar_offaxis(s)
        mocap, mocap_epoch = {}, {}
        for r in rules:
            mocap[r] = in_window(mocap_transitions(s, r), s.t, w)
            mocap_epoch[r] = epoch_posture(r.classify(s.mocap), N)
        speeds = {k: s.radar[k] for k in ('v_peak',) + PEAK_FEATURES}
        return dict(rec=rec, person=s.person, t=s.t, window=w, height=h, offaxis=off, speeds=speeds,
                    valid_epoch=epoch_mean(w, N) >= 0.5, bursts=b, mocap=mocap, mocap_epoch=mocap_epoch), None
    except Exception:
        return None, f'{rec}: {traceback.format_exc().strip().splitlines()[-1]}'


DATA, RULE_OF, SCORED = {}, {}, set()   # filled before the worker pools fork


def keep(r):
    """the events of recording r inside its evaluation window"""
    return lambda e: in_window(e, DATA[r]['t'], DATA[r]['window'])


# ---- main method: posture from height ----

def radar_events(r, rule, posture=None, speed='v_peak'):
    """the radar transitions of recording r inside its window; peak speed from the per-frame speed `speed`"""
    d = DATA[r]
    if posture is None:
        posture = rule.classify(d['height'], d['offaxis'])
    return keep(r)(radar_posture_transitions(d['t'], posture, d['speeds'][speed], rule.min_hold_s))


# ---- peak Doppler: which speed measure ----

def peak_pairs(recs, measure):
    """radar peak (measure, within 1 s) and MoCap peak at the counted MoCap transitions of recs, under each
    recording's person's MoCap rule"""
    rows = []
    for r in recs:
        d = DATA[r]
        m = d['mocap'][RULE_OF[d['person']]]
        m = m[m.kind.isin(COUNTED)]
        radar = add_peak(m[['t']].assign(peak_v=np.nan), d['t'], d['speeds'][measure]).peak_v.values
        rows.append(pd.DataFrame(dict(rec=r, person=d['person'], radar=radar, mocap=m.peak_v.values)))
    return pd.concat(rows, ignore_index=True).dropna()


def calibrate_peak(person):
    """the speed quantile whose peaks are closest to MoCap's (mean absolute error) on every person but `person`"""
    train = [r for r in DATA if DATA[r]['person'] != person]
    mae = {k: (lambda g: (g.radar - g.mocap).abs().mean())(peak_pairs(train, k)) for k in PEAK_FEATURES}
    best = min(mae, key=mae.get)
    return best, mae[best]


def height_epochs(r, rule):
    """the corrected radar height per epoch of recording r"""
    return epoch_mean(rule.height(DATA[r]['height'], DATA[r]['offaxis']), N)


def calibrate_height_rule(person):
    """the RadarPostureRule calibrated on every person but `person` ('ALL' when None), with MoCap under that person's
    MoCap posture rule: per off-axis correction, thresholds per epoch; then the correction and hold with the best
    transition F1. Ties: the smallest correction and hold."""
    mrule = RULE_OF.get(person, RULE_OF['ALL'])
    train = [r for r in DATA if DATA[r]['person'] != person]
    best = None
    for k in GRID_OFFAXIS_K:
        ep = []
        for r in train:
            h, mp = height_epochs(r, RadarPostureRule(0, 0, 0, 0, k)), DATA[r]['mocap_epoch'][mrule]
            E = min(len(h), len(mp))
            keep_ = DATA[r]['valid_epoch'][:E] & (mp[:E] != UNKNOWN)
            ep.append(pd.DataFrame(dict(person=DATA[r]['person'], height=h[:E], posture=mp[:E]))[keep_])
        th = calibrate_radar_thresholds(pd.concat(ep, ignore_index=True))
        postures = {r: RadarPostureRule(*th, 0, k).classify(DATA[r]['height'], DATA[r]['offaxis']) for r in train}
        for hold in GRID_MIN_HOLD_S:
            rule = RadarPostureRule(*th, hold, k)
            hit = n = 0
            for r in train:
                radar, mocap = radar_events(r, rule, postures[r]), DATA[r]['mocap'][mrule]
                hit += len(match_events(mocap, radar))
                n += len(mocap) + len(radar)
            f1 = 2 * hit / n if n else 0.0
            if best is None or f1 > best[1]:
                best = (rule, f1)
    return best


# ---- comparison: burst classifiers ----

def fit(recs, rule, features):
    """burst classifier trained on the in-window bursts of recs, targets from MoCap under the posture rule"""
    b = pd.concat([DATA[r]['bursts'][DATA[r]['bursts'].in_window] for r in recs], ignore_index=True)
    y = np.concatenate([burst_targets(DATA[r]['bursts'][DATA[r]['bursts'].in_window], DATA[r]['mocap'][rule])
                        for r in recs])
    return BurstModel(features).fit(b, y), len(b), int((y != 'none').sum())


def run_fold(job):
    """one left-out person and classifier, in a worker: detection threshold from out-of-fold classifications of the
    other persons, classifier trained on all of them, transitions of the person's recordings"""
    person, method = job
    features, rule = CLASSIFIERS[method], RULE_OF[person]
    others = sorted({d['person'] for d in DATA.values()} - {person})
    train = [r for r in DATA if DATA[r]['person'] != person]
    oof = []
    for k in range(INNER_FOLDS):
        held = set(others[k::INNER_FOLDS])
        m, _, _ = fit([r for r in train if DATA[r]['person'] not in held], rule, features)
        oof += [(m.classify(DATA[r]['bursts']), DATA[r]['mocap'][rule], keep(r))
                for r in train if DATA[r]['person'] in held]
    min_prob, train_f1 = calibrate_min_prob(oof)
    model, _, _ = fit(train, rule, features)
    out = [(r, keep(r)(radar_transitions(model.classify(DATA[r]['bursts']), min_prob)))
           for r in DATA if DATA[r]['person'] == person and r in SCORED]
    return dict(person=person, method=method, min_prob=min_prob, train_f1=train_f1), out


def rule_events(b, v_min, rise_min, dur_min):
    """hand-rule baseline: bursts passing the three thresholds, one per direction within NMS_S (the fastest)"""
    k = b[(b.v_smooth >= v_min) & (b.rise >= rise_min) & (b.duration >= dur_min)].sort_values('v_smooth',
                                                                                              ascending=False)
    kept = []
    for x in k.itertuples():
        if not any(abs(x.t - y.t) < NMS_S and x.direction == y.direction for y in kept):
            kept.append(x)
    return pd.DataFrame([dict(t=x.t, kind='transition', direction=x.direction) for x in kept],
                        columns=['t', 'kind', 'direction'])


def metrics_of(job):
    """the Category 3 metrics of one recording, in a worker"""
    th, radar, mocap, postures = job
    try:
        s = load_session(*parse_recording(th.rec))
        return ev.metric_table([s], {s.name: th.radar_threshold}, {s.person: th.mocap_threshold}, category=3,
                               transitions={s.name: (radar, mocap)}, postures={s.name: postures},
                               active_feature=th.active_feature, energy_feature=th.energy_feature)
    except Exception:
        print(f'{th.rec}: metrics FAILED: {traceback.format_exc().strip().splitlines()[-1]}', flush=True)
        return None


def spread(x, fmt='.3f'):
    """median [10th-90th percentile]"""
    q = np.nanpercentile(x, [50, 10, 90])
    return f'{q[0]:{fmt}} [{q[1]:{fmt}}-{q[2]:{fmt}}]'


def per_kind_table(events):
    rows = []
    for k in sorted(set(events.kind)):
        m, r = events[(events.source == 'mocap') & (events.kind == k)], events[(events.source == 'radar')
                                                                               & (events.kind == k)]
        rows.append(dict(kind=k, mocap=len(m), radar=len(r), matched=int(m.matched.sum()),
                         f1=2 * m.matched.sum() / (len(m) + len(r)) if len(m) + len(r) else np.nan,
                         **{f'as_{o or "missed"}': int((m.other_kind == o).sum()) for o in sorted(set(m.other_kind))}))
    return pd.DataFrame(rows).fillna(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO', help='recordings to score (all calibrate)')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    ap.add_argument('--fold-workers', type=int, default=13, help='left-out persons trained in parallel')
    args = ap.parse_args()
    th = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv'))   # recordings evaluated in Category 1
    rules = posture_rules()
    RULE_OF.update({p: rules.get(p, rules['ALL']) for p in th.person.unique()}, ALL=rules['ALL'])
    out = os.path.join(OUTPUT_ROOT, 'category3')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)

    jobs = [(r, sorted(set(RULE_OF.values()), key=str)) for r in th.rec]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(prepare, jobs)
    else:
        res = [prepare(j) for j in jobs]
    failed = [e for _, e in res if e]
    if failed:
        print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
    DATA.update({d['rec']: d for d, e in res if not e})
    SCORED.update(args.only if args.only else DATA)
    persons = sorted({d['person'] for d in DATA.values()})
    seen = np.mean(np.concatenate([~np.isnan(d['height'][d['window']]) for d in DATA.values()]))
    print(f'{len(DATA)} recordings, {len(persons)} persons; radar sees the person in {100 * seen:.0f}% of the '
          f'evaluated frames', flush=True)

    # ---- main method: height rule, leave-one-subject-out ----
    rule_rows, height_rule = [], {}
    for person in persons + [None]:
        rule, f1 = calibrate_height_rule(person)
        height_rule[person or 'ALL'] = rule
        rule_rows.append(dict(person=person or 'ALL', h_stand=rule.h_stand, h_sit=rule.h_sit, h_floor=rule.h_floor,
                              min_hold_s=rule.min_hold_s, offaxis_k=rule.offaxis_k, train_transition_f1=f1))
    rr = pd.DataFrame(rule_rows)
    rr.to_csv(RADAR_POSTURE_RULE_FILE, index=False)
    held = rr[rr.person != 'ALL']
    print(f'height rule, all persons: {height_rule["ALL"]}\n  leaving each person out: standing > '
          f'{held.h_stand.min():.2f}-{held.h_stand.max():.2f} m, sitting > {held.h_sit.min():.2f}-'
          f'{held.h_sit.max():.2f} m, lying > {held.h_floor.min():.2f}-{held.h_floor.max():.2f} m, hold '
          f'{sorted(held.min_hold_s.unique())} s, off-axis correction {sorted(held.offaxis_k.unique())} m/m; '
          f'training transition F1 {held.train_transition_f1.min():.3f}-{held.train_transition_f1.max():.3f}',
          flush=True)

    # ---- peak Doppler: the speed measure, leave-one-subject-out ----
    pairs = {k: peak_pairs(list(DATA), k) for k in ('v_peak',) + PEAK_FEATURES}
    rows = []
    for k, g in pairs.items():
        diff = g.radar - g.mocap
        loso = g.person.map({p: diff[g.person != p].mean() for p in g.person.unique()})
        rows.append(dict(measure=k, transitions=len(g), radar_median=g.radar.median(), mocap_median=g.mocap.median(),
                         bias=diff.mean(), mae=diff.abs().mean(), mae_bias_removed=(diff - loso).abs().mean()))
    pm = pd.DataFrame(rows)
    pm.to_csv(f'{out}/peak_measures.csv', index=False)
    print(f'\npeak radial speed within 1 s of the counted MoCap transitions, radar measure vs MoCap fastest joint:\n'
          f'{pm.round(3).to_string(index=False)}')
    peak_rule = {}
    for person in persons + [None]:
        peak_rule[person or 'ALL'] = calibrate_peak(person)
    pr = pd.DataFrame([dict(person=p, measure=m, train_mae=e) for p, (m, e) in peak_rule.items()])
    pr.to_csv(f'{out}/peak_rule.csv', index=False)
    print(f'peak measure, leaving each person out: {pr[pr.person != "ALL"].measure.value_counts().to_dict()}; '
          f'all persons: {peak_rule["ALL"][0]}', flush=True)

    transitions, postures, posture_rows, confusion, scores, events = {}, {}, [], [], [], []
    for r in sorted(SCORED):
        d = DATA[r]
        rule, mrule = height_rule[d['person']], RULE_OF[d['person']]
        frames = rule.classify(d['height'], d['offaxis'])
        radar, mocap = radar_events(r, rule, frames, peak_rule[d['person']][0]), d['mocap'][mrule]
        rp, mp = epoch_posture(frames, N), d['mocap_epoch'][mrule]
        E = min(len(rp), len(mp), len(d['valid_epoch']))
        transitions[r], postures[r] = (radar, mocap), (rp, mp)
        posture_rows.append(dict(rec=r, **ev.posture_scores(rp[:E], mp[:E], height_epochs(r, rule)[:E],
                                                            d['valid_epoch'][:E])))
        v = d['valid_epoch'][:E] & (mp[:E] != UNKNOWN)
        confusion.append(pd.crosstab(pd.Series(mp[:E][v], name='mocap'), pd.Series(rp[:E][v], name='radar')))
        scores.append(dict(rec=r, method='height rule', **ev.transition_scores(radar, mocap)))
        events.append(ev.transition_events(radar, mocap, d['t'][0]).assign(rec=r))

    ps = pd.DataFrame(posture_rows)
    ps.to_csv(f'{out}/posture_scores.csv', index=False)
    conf = sum(c.reindex(index=range(4), columns=[UNKNOWN, 0, 1, 2, 3], fill_value=0) for c in confusion)
    conf.index = [POSTURES[i] for i in conf.index]
    conf.columns = ['unknown'] + list(POSTURES)
    conf.to_csv(f'{out}/posture_confusion.csv')
    print('\nposture per 1 s epoch (MoCap as reference), median [10th-90th percentile] over recordings:')
    print(f'  radar sees the person in {spread(ps.seen, ".2f")} of the epochs')
    for c in [c for c in ps.columns if c.startswith(('f1_', 'auc_'))]:
        print(f'  {c:24s} {spread(ps[c])}')
    print(f'pooled epochs, MoCap (rows) x radar (columns):\n{conf.to_string()}')

    # ---- comparisons: classifiers and the hand rule ----
    with Pool(args.fold_workers) as pool:
        folds = pool.map(run_fold, [(p, m) for m in CLASSIFIERS for p in persons])
    pd.DataFrame([f for f, _ in folds]).to_csv(f'{out}/thresholds.csv', index=False)
    for f, recs in folds:
        for r, radar in recs:
            scores.append(dict(rec=r, method=f['method'], **ev.transition_scores(radar, transitions[r][1])))
    base_counts, base_rows = {}, []
    for d in DATA.values():
        mocap = d['mocap'][RULE_OF[d['person']]]
        for g, params in enumerate(RULE_GRID):
            r = in_window(rule_events(d['bursts'], *params), d['t'], d['window'])
            base_counts[d['rec'], g] = (len(match_events(mocap, r, typed=False)), len(r), len(mocap))
    for person in persons:
        f1 = [2 * sum(c[0] for (rec, gg), c in base_counts.items() if gg == g and DATA[rec]['person'] != person)
              / sum(c[1] + c[2] for (rec, gg), c in base_counts.items() if gg == g and DATA[rec]['person'] != person)
              for g in range(len(RULE_GRID))]
        g = int(np.argmax(f1))
        for rec in (r for r in DATA if DATA[r]['person'] == person and r in SCORED):
            hit, n_r, n_m = base_counts[rec, g]
            base_rows.append(dict(rec=rec, v_min=RULE_GRID[g][0], rise_min=RULE_GRID[g][1], dur_min=RULE_GRID[g][2],
                                  mocap_events=n_m, radar_events=n_r, matched_any=hit,
                                  f1_any=2 * hit / (n_m + n_r) if n_m + n_r else np.nan))
    base = pd.DataFrame(base_rows)
    base.to_csv(f'{out}/rule_baseline.csv', index=False)

    # ---- per event ----
    sc = pd.DataFrame(scores)
    sc.to_csv(f'{out}/methods.csv', index=False)
    main = sc[sc.method == 'height rule'].drop(columns='method')
    main.to_csv(f'{out}/event_scores.csv', index=False)
    events = pd.concat(events, ignore_index=True)
    events.to_csv(f'{out}/events.csv', index=False)
    print('\nper transition (MoCap as reference, +-1 s), median [10th-90th percentile] over recordings:')
    for m, g in sc.groupby('method', sort=False):
        print(f'  {m:17s} same kind F1 {spread(g.f1)}, precision {spread(g.precision)}, recall {spread(g.recall)}; '
              f'any kind F1 {spread(g.f1_any)}; pooled {g.matched.sum()} matched of {g.mocap_events.sum()} MoCap, '
              f'{g.radar_events.sum()} radar')
    print(f'  {"hand rule":17s} any kind F1 {spread(base.f1_any)}')
    print(f'  height rule, matched: time error {spread(main.time_mae_s, ".2f")} s, peak speed error '
          f'{spread(main.peak_mae, ".2f")} m/s (per-recording MAE)')
    per_kind = per_kind_table(events)
    per_kind.to_csv(f'{out}/per_kind.csv', index=False)
    print(f'\nheight rule, per kind, pooled (as_<kind>: radar kind of the MoCap transitions, by time only):\n'
          f'{per_kind.round(3).to_string(index=False)}')
    m = events[(events.source == 'mocap') & events.kind.isin(COUNTED)]
    off = m.groupby(pd.cut(m.offaxis_m, [0, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0])).matched.agg(['size', 'mean'])
    off.to_csv(f'{out}/offaxis.csv')
    print(f'\ncounted MoCap transitions found (same kind), by distance from the radar axis (m):\n'
          f'{off.round(3).to_string()}')

    # ---- per recording: the metrics ----
    mjobs = [(r, *transitions[r.rec], postures[r.rec]) for _, r in th.iterrows() if r.rec in transitions]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            mts = pool.map(metrics_of, mjobs)
    else:
        mts = [metrics_of(j) for j in mjobs]
    mt = pd.concat([x for x in mts if x is not None], ignore_index=True)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    agreement = ev.metric_agreement(mt)
    person = mt.rec.str.split('/').str[0]
    for k, g in mt.groupby('metric', sort=False):
        i = agreement.metric == k
        agreement.loc[i, 'mocap_median'] = g.mocap.median()
        agreement.loc[i, 'radar_median'] = g.radar.median()
        agreement.loc[i, 'rel_mae'] = (g.radar - g.mocap).abs().mean() / g.mocap.abs().mean()
        # a constant offset could be calibrated away: the error left after removing the other persons' mean bias
        diff, who = g.radar - g.mocap, person[g.index]
        agreement.loc[i, 'mae_bias_removed'] = np.nanmean(np.abs(diff - who.map(
            {q: diff[who != q].mean() for q in who.unique()})))
        # no sensor at all: the median MoCap value of the other persons. The script fixes how many transitions of
        # each kind there are, and roughly how long each posture lasts, so the radar is informative where it beats this
        p = person[g.index]
        b_ = np.array([g.mocap[p != q].median() for q in p])
        agreement.loc[i, 'baseline_mae'] = np.nanmean(np.abs(b_ - g.mocap.values))
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(f'\nper-recording metrics (height rule):\n{agreement.round(3).to_string(index=False)}')
    for k, g in mt.groupby('metric', sort=False):
        print(f'  {k:24s} MoCap {spread(g.mocap, ".2f")}  radar {spread(g.radar, ".2f")}')
