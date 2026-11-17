"""
Category 3 (postural transitions, time in each posture): radar vs MoCap on the recordings evaluated in Category 1.
Run first:
    scripts/evaluate_category1.py     (thresholds.csv: the recordings, the evaluation window)
    scripts/calibrate_posture.py      (MoCap posture rule per left-out person)
    scripts/angle_pass.py             (points.npz: the angle; the person's height per frame is derived from it)

    python scripts/evaluate_category3.py --workers 16
    python scripts/evaluate_category3.py --only person_1/scenario_1 person_5/scenario_3   (calibrates on all)

The calibration and scoring are in mobility/evaluation/category3.py; this script runs them and writes the tables.

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
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '6')    # threads per classifier; folds run in parallel (--fold-workers)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.cli import error_summary, parallel_map, wide_tables  # noqa: E402
from mobility.config import CATEGORY1_THRESHOLDS, EPOCH_FRAMES, category_dir  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.posture import POSTURES, RADAR_POSTURE_RULE_FILE, UNKNOWN, epoch_posture, posture_rules  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.transitions import COUNTED  # noqa: E402
from mobility.utils import fold_of, spread  # noqa: E402

# filled before the worker pools fork: prepared recordings (mobility.evaluation.prepare_recording), the MoCap posture
# rule per person ('ALL' too), and the recordings to score (--only)
DATA, RULE_OF, SCORED = {}, {}, set()


def prepare(job):
    """one recording, in a worker: radar height, bursts, and the MoCap posture and transitions under every MoCap
    posture rule a fold may use"""
    rec, rules = job
    try:
        return ev.prepare_recording(rec, rules), None
    except Exception:
        return None, f'{rec}: {error_summary()}'


def run_fold(job):
    """one left-out person and classifier, in a worker (mobility.evaluation.burst_classifier_fold)"""
    person, method = job
    return ev.burst_classifier_fold(DATA, RULE_OF, SCORED, person, method)


def metrics_of(job):
    """the Category 3 metrics of one recording, in a worker"""
    th, radar, mocap, postures = job
    try:
        s = load_session(*parse_recording(th.rec))
        return ev.metric_table([s], {s.name: th.radar_threshold}, {s.person: th.mocap_threshold}, category=3,
                               transitions={s.name: (radar, mocap)}, postures={s.name: postures},
                               active_feature=th.active_feature, energy_feature=th.energy_feature)
    except Exception:
        print(f'{th.rec}: metrics FAILED: {error_summary()}', flush=True)
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO', help='recordings to score (all calibrate)')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    ap.add_argument('--fold-workers', type=int, default=13, help='left-out persons trained in parallel')
    args = ap.parse_args()
    th = pd.read_csv(CATEGORY1_THRESHOLDS)   # recordings evaluated in Category 1
    rules = posture_rules()
    RULE_OF.update({p: fold_of(rules, p) for p in th.person.unique()}, ALL=rules['ALL'])
    out = category_dir(3)
    os.makedirs(out, exist_ok=True)
    wide_tables(max_columns=30)

    res = parallel_map(prepare, [(r, sorted(set(RULE_OF.values()), key=str)) for r in th.rec], args.workers)
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
        rule, f1 = ev.calibrate_height_rule(DATA, RULE_OF, person)
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
    pm = ev.peak_measures(DATA, RULE_OF)
    pm.to_csv(f'{out}/peak_measures.csv', index=False)
    print(f'\npeak radial speed within 1 s of the counted MoCap transitions, radar measure vs MoCap fastest joint:\n'
          f'{pm.round(3).to_string(index=False)}')
    peak_rule = {}
    for person in persons + [None]:
        peak_rule[person or 'ALL'] = ev.calibrate_peak(DATA, RULE_OF, person)
    pr = pd.DataFrame([dict(person=p, measure=m, train_mae=e) for p, (m, e) in peak_rule.items()])
    pr.to_csv(f'{out}/peak_rule.csv', index=False)
    print(f'peak measure, leaving each person out: {pr[pr.person != "ALL"].measure.value_counts().to_dict()}; '
          f'all persons: {peak_rule["ALL"][0]}', flush=True)

    # ---- main method, per recording: posture per epoch and transitions ----
    transitions, postures, posture_rows, confusion, scores, events = {}, {}, [], [], [], []
    for r in sorted(SCORED):
        d = DATA[r]
        rule, mrule = height_rule[d['person']], RULE_OF[d['person']]
        frames = rule.classify(d['height'], d['offaxis'])
        radar, mocap = ev.radar_events(d, rule, frames, peak_rule[d['person']][0]), d['mocap'][mrule]
        rp, mp = epoch_posture(frames, EPOCH_FRAMES), d['mocap_epoch'][mrule]
        E = min(len(rp), len(mp), len(d['valid_epoch']))
        transitions[r], postures[r] = (radar, mocap), (rp, mp)
        posture_rows.append(dict(rec=r, **ev.posture_scores(rp[:E], mp[:E], ev.height_epochs(d, rule)[:E],
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
    folds = parallel_map(run_fold, [(p, m) for m in ev.CLASSIFIERS for p in persons], args.fold_workers)
    pd.DataFrame([f for f, _ in folds]).to_csv(f'{out}/thresholds.csv', index=False)
    for f, recs in folds:
        for r, radar in recs:
            scores.append(dict(rec=r, method=f['method'], **ev.transition_scores(radar, transitions[r][1])))
    base = ev.hand_rule_baseline(DATA, RULE_OF, SCORED)
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
    per_kind = ev.per_kind_table(events)
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
    mts = parallel_map(metrics_of, mjobs, args.workers)
    mt = pd.concat([x for x in mts if x is not None], ignore_index=True)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    # no sensor at all: the median MoCap value of the other persons. The script fixes how many transitions of each
    # kind there are, and roughly how long each posture lasts, so the radar is informative where it beats this. A
    # constant offset could be calibrated away: mae_bias_removed, after removing the other persons' mean bias
    agreement = ev.agreement_with_baseline(mt, bias_removed=True)
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(f'\nper-recording metrics (height rule):\n{agreement.round(3).to_string(index=False)}')
    for k, g in mt.groupby('metric', sort=False):
        print(f'  {k:24s} MoCap {spread(g.mocap, ".2f")}  radar {spread(g.radar, ".2f")}')
