"""
Category 2 (activity bout structure): radar vs MoCap on the recordings evaluated in Category 1. Bouts come from the
Category 1 active / inactive calls, so this reuses its leave-one-subject-out thresholds. Run first:
    scripts/evaluate_category1.py     (thresholds.csv: radar and MoCap active thresholds per recording)
    scripts/calibrate_bout_rule.py    (bout rule per left-out person)

    python scripts/evaluate_category2.py --workers 16
    python scripts/evaluate_category2.py --only person_1/scenario_1 person_5/scenario_3

Writes to OUTPUT_ROOT/category2/:
    rules.csv          per recording: active thresholds and the bout rule used for it
    epoch_scores.csv   per recording: F1 of the radar's active calls and of its in-bout calls (MoCap as reference)
    bout_scores.csv    per recording: bouts of each sensor, matched bouts, precision, recall, F1, onset / offset error
    bouts.csv          every bout of both sensors, matched or not
    metrics.csv        per recording: the Category 2 metrics, radar vs MoCap
    agreement.csv      per metric across recordings: mean absolute error and bias
    sensitivity.csv    the same agreement and bout F1 for fixed rules other than the calibrated one
"""
import argparse
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.bouts import BoutRule, bout_rules  # noqa: E402
from mobility.config import EPOCH_S, OUTPUT_ROOT  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.session import load_session  # noqa: E402

# fixed rules (epochs) for the sensitivity check: how much the radar-MoCap agreement depends on the rule
SENSITIVITY = [BoutRule(g, b) for g in (0, 1, 2, 5) for b in (1, 2, 3, 5)]


def evaluate(job):
    """one recording, in a worker: bouts, scores and metrics with the calibrated rule, and the sensitivity rules"""
    th, rule = job
    try:
        s = load_session(*parse_recording(th.rec))
        radar_kw = dict(active_feature=th.active_feature, energy_feature=th.energy_feature)
        pairs = ev.epoch_pairs(s, th.radar_threshold, th.mocap_threshold, **radar_kw)

        def run(r):
            p = ev.add_bouts(pairs, r)
            events = ev.bout_events(p)
            mt = ev.metric_table([s], {s.name: th.radar_threshold}, {s.person: th.mocap_threshold}, category=2,
                                 bout_rules={s.person: r}, **radar_kw)
            return events.assign(rec=s.name), dict(rec=s.name, **ev.bout_scores(p, events)), mt

        events, scores, mt = run(rule)
        sens = []
        for r in SENSITIVITY:
            _, sc, m = run(r)
            sens.append(m.assign(max_gap_s=r.max_gap * EPOCH_S, min_bout_s=r.min_bout * EPOCH_S,
                                 bout_f1=sc['bout_f1'], f1_bout_epochs=sc['f1_bout_epochs']))
        return events, scores, mt, pd.concat(sens), None
    except Exception:
        return None, None, None, None, f'{th.rec}: {traceback.format_exc().strip().splitlines()[-1]}'


def spread(x, fmt='.3f'):
    """median [10th-90th percentile]"""
    q = np.nanpercentile(x, [50, 10, 90])
    return f'{q[0]:{fmt}} [{q[1]:{fmt}}-{q[2]:{fmt}}]'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--workers', type=int, default=1, help='recordings evaluated in parallel')
    args = ap.parse_args()
    th = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv'))   # recordings evaluated in Category 1
    if args.only:
        th = th[th.rec.isin(args.only)]
    rules = bout_rules()
    out = os.path.join(OUTPUT_ROOT, 'category2')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    jobs = [(r, rules.get(r.person, rules['ALL'])) for _, r in th.iterrows()]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(evaluate, jobs)
    else:
        res = [evaluate(j) for j in jobs]
    failed = [r[-1] for r in res if r[-1]]
    if failed:
        print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
    res = [r for r in res if not r[-1]]
    ok = {sc['rec'] for _, sc, _, _, _ in res}
    print(f'{len(res)} recordings, {th[th.rec.isin(ok)].person.nunique()} persons')

    used = pd.DataFrame([dict(rec=r.rec, person=r.person, radar_threshold=r.radar_threshold,
                              mocap_threshold=r.mocap_threshold, max_gap_s=rule.max_gap * EPOCH_S,
                              min_bout_s=rule.min_bout * EPOCH_S) for r, rule in jobs if r.rec in ok])
    used.to_csv(f'{out}/rules.csv', index=False)
    print('bout rules used:', used.groupby(['max_gap_s', 'min_bout_s']).size().to_dict())

    events = pd.concat([r[0] for r in res], ignore_index=True)
    events.to_csv(f'{out}/bouts.csv', index=False)
    scores = pd.DataFrame([r[1] for r in res])
    scores[['rec', 'epochs', 'f1_active', 'f1_bout_epochs']].to_csv(f'{out}/epoch_scores.csv', index=False)
    scores.drop(columns=['f1_active', 'f1_bout_epochs']).to_csv(f'{out}/bout_scores.csv', index=False)
    print(f'\nper epoch, F1 over recordings, median [10th-90th percentile]: active calls {spread(scores.f1_active)}, '
          f'in-bout calls {spread(scores.f1_bout_epochs)}')
    print(f'per bout (IoU >= {ev.BOUT_MATCH_IOU}): F1 {spread(scores.bout_f1)}, precision '
          f'{spread(scores.bout_precision)}, recall {spread(scores.bout_recall)}')
    print(f'  pooled: {scores.matched.sum()} matched of {scores.mocap_bouts.sum()} MoCap and '
          f'{scores.radar_bouts.sum()} radar bouts; matched bouts: onset error {spread(scores.onset_mae_s, ".2f")} s, '
          f'offset error {spread(scores.offset_mae_s, ".2f")} s (per-recording MAE)')
    m = events.dropna(subset=['onset_err_s'])
    within = ((m.onset_err_s.abs() <= 1) & (m.offset_err_s.abs() <= 1)).mean()
    print(f'  pooled over matched bouts: mean onset error {m.onset_err_s.mean():+.2f} s, offset '
          f'{m.offset_err_s.mean():+.2f} s (radar - MoCap); onset and offset both within 1 s: {within:.2f}')
    print(f'lowest bout F1:\n{scores.nsmallest(8, "bout_f1").round(3).to_string(index=False)}')
    for src, g in events.groupby('source'):
        print(f'  {src} bout duration (s), all bouts: {spread(g.duration_s, ".1f")}, '
              f'matched {g.matched.mean():.2f}; unmatched bouts {spread(g[~g.matched].duration_s, ".1f")}')

    mt = pd.concat([r[2] for r in res], ignore_index=True)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    agreement = ev.metric_agreement(mt)
    person = mt.rec.str.split('/').str[0]
    for k, g in mt.groupby('metric', sort=False):
        i = agreement.metric == k
        agreement.loc[i, 'mocap_median'] = g.mocap.median()
        agreement.loc[i, 'radar_median'] = g.radar.median()
        agreement.loc[i, 'rel_mae'] = (g.radar - g.mocap).abs().mean() / g.mocap.abs().mean()
        # no sensor at all: every recording gets the median MoCap value of the other persons. The script imposes
        # much of the bout structure, so the radar is only useful where it beats this.
        p = person[g.index]
        base = np.array([g.mocap[p != q].median() for q in p])
        agreement.loc[i, 'baseline_mae'] = np.nanmean(np.abs(base - g.mocap.values))
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(f'\nper-recording metrics:\n{agreement.round(4).to_string(index=False)}')
    for k, g in mt.groupby('metric', sort=False):
        per_person = (g.radar - g.mocap).groupby(g.rec.str.split('/').str[0]).mean()
        print(f'  {k:20s} MoCap {spread(g.mocap)}  radar {spread(g.radar)}  mean error per person '
              f'{per_person.min():+.3f} ({per_person.idxmin()}) to {per_person.max():+.3f} ({per_person.idxmax()})')

    sens = pd.concat([r[3] for r in res], ignore_index=True)
    rows = []
    for (gap, mb, k), g in sens.groupby(['max_gap_s', 'min_bout_s', 'metric'], sort=False):
        d = g.radar - g.mocap
        rows.append(dict(max_gap_s=gap, min_bout_s=mb, metric=k, mocap_median=g.mocap.median(),
                         mean_abs_err=d.abs().mean(), bias=d.mean(), rel_mae=d.abs().mean() / g.mocap.abs().mean(),
                         bout_f1=g.bout_f1.median(), f1_bout_epochs=g.f1_bout_epochs.median()))
    sens = pd.DataFrame(rows)
    sens.to_csv(f'{out}/sensitivity.csv', index=False)
    print(f'\nsensitivity to the rule (relative MAE = MAE / mean MoCap value; F1 medians over recordings):\n'
          f'{sens.pivot_table(index=["max_gap_s", "min_bout_s"], columns="metric", values="rel_mae").round(3)}\n'
          f'{sens.groupby(["max_gap_s", "min_bout_s"])[["bout_f1", "f1_bout_epochs"]].first().round(3)}')
