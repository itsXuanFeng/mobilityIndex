"""
Category 2 (activity bout structure): radar vs MoCap on the recordings evaluated in Category 1. Bouts start from the
Category 1 active / inactive calls, so this reuses its leave-one-subject-out thresholds as the entry thresholds of the
hysteresis. Run first:
    scripts/evaluate_category1.py     (thresholds.csv: radar and MoCap active thresholds per recording)
    scripts/calibrate_bout_rule.py    (MoCap exit and confirmation factors and bout rule per left-out person)
The radar exit and confirmation thresholds are calibrated here, against the MoCap bout counts, leaving the person out.

    python scripts/evaluate_category2.py --workers 16
    python scripts/evaluate_category2.py --only person_1/scenario_1 person_5/scenario_3

Writes to OUTPUT_ROOT/category2/:
    rules.csv              per recording: entry, exit and confirmation thresholds of both sensors, the bout rule
    radar_levels.csv       per left-out person and radar (exit delta, confirmation eps): training count error, bout F1
    epoch_scores.csv       per recording: F1 of the radar's active calls and of its in-bout calls (MoCap as reference)
    bout_scores.csv        per recording: bouts of each sensor, matched, precision, recall, F1, onset / offset error
    bouts.csv              every bout of both sensors, matched or not, with the activity label at its middle
    metrics.csv            per recording: the Category 2 metrics, radar vs MoCap
    agreement.csv          per metric across recordings: mean absolute error, bias, no-sensor baseline
    sensitivity.csv        the same agreement and bout F1 for the radar calibrated on bout F1, for fixed radar
                           thresholds, and without hysteresis and confirmation
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
from mobility.bouts import bout_rules, mocap_level_factors  # noqa: E402
from mobility.config import EPOCH_S, FPS, OUTPUT_ROOT  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.session import load_session  # noqa: E402


def prepare(th):
    """one recording, in a worker: radar and MoCap States on the same valid epochs, and the label per epoch"""
    try:
        s = load_session(*parse_recording(th.rec))
        rs, ms = ev.paired_states(s, th.radar_threshold, th.mocap_threshold, active_feature=th.active_feature,
                                  energy_feature=th.energy_feature)
        n = int(round(EPOCH_S * FPS))
        return th.rec, th.person, rs, ms, s.label_per_frame()[n // 2::n][:len(rs.t)], None
    except Exception:
        return th.rec, th.person, None, None, None, f'{th.rec}: {traceback.format_exc().strip().splitlines()[-1]}'


def evaluate(rec, rs, ms, labels, rule, levels):
    """bouts, scores and metrics of one recording with the given bout rule and levels: ((radar exit, radar
    confirmation), (MoCap exit, MoCap confirmation)) thresholds"""
    rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
    ev.set_bout_levels(rs, ms, *levels)
    frame = ev.bout_frame(rs, ms)
    events = ev.bout_events(frame, labels=labels).assign(rec=rec)
    return events, dict(rec=rec, **ev.bout_scores(frame, events)), pd.DataFrame(ev.metric_rows(rec, rs, ms, 2))


def agreement_table(mt):
    """per metric: MAE and bias (radar - MoCap), medians, MAE relative to the mean MoCap value, and the MAE of a guess
    without any sensor: every recording gets the median MoCap value of the other persons. The script imposes much of
    the bout structure, so the radar is only useful where it beats this guess."""
    agreement = ev.metric_agreement(mt)
    person = mt.rec.str.split('/').str[0]
    for k, g in mt.groupby('metric', sort=False):
        i = agreement.metric == k
        p = person[g.index]
        base = np.array([g.mocap[p != q].median() for q in p])
        agreement.loc[i, 'mocap_median'] = g.mocap.median()
        agreement.loc[i, 'radar_median'] = g.radar.median()
        agreement.loc[i, 'rel_mae'] = (g.radar - g.mocap).abs().mean() / g.mocap.abs().mean()
        agreement.loc[i, 'baseline_mae'] = np.nanmean(np.abs(base - g.mocap.values))
    agreement['vs_baseline'] = agreement.mean_abs_err / agreement.baseline_mae
    return agreement


def spread(x, fmt='.3f'):
    """median [10th-90th percentile]"""
    q = np.nanpercentile(x, [50, 10, 90])
    return f'{q[0]:{fmt}} [{q[1]:{fmt}}-{q[2]:{fmt}}]'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    args = ap.parse_args()
    th = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv'))   # recordings evaluated in Category 1
    if args.only:
        th = th[th.rec.isin(args.only)]
    rows = [r for _, r in th.iterrows()]
    out = os.path.join(OUTPUT_ROOT, 'category2')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(prepare, rows)
    else:
        res = [prepare(r) for r in rows]
    failed = [r[-1] for r in res if r[-1]]
    if failed:
        print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
    res = [r for r in res if not r[-1]]
    th = th[th.rec.isin({r[0] for r in res})].set_index('rec')
    print(f'{len(res)} recordings, {th.person.nunique()} persons', flush=True)

    # definitions, all calibrated without the evaluated person
    rules, factors = bout_rules(), mocap_level_factors()
    rule_of = {p: rules.get(p, rules['ALL']) for p in th.person.unique()}
    factor_of = {p: factors.get(p, factors['ALL']) for p in th.person.unique()}
    radar_thr = th.groupby('person').radar_threshold.first().to_dict()
    mocap_thr = th.groupby('person').mocap_threshold.first().to_dict()
    states = [(p, rs, ms) for _, p, rs, ms, _, _ in res]
    radar_of, radar_table = ev.loso_radar_bout_levels(states, radar_thr, mocap_thr, factor_of, rule_of)
    radar_table.to_csv(f'{out}/radar_levels.csv', index=False)

    def levels(rec, p, delta, eps):
        """((radar exit, confirmation), (MoCap exit, confirmation)) thresholds of a recording of person p"""
        r, m = th.radar_threshold[rec], th.mocap_threshold[rec]
        return (r - delta, r + eps), (factor_of[p][0] * m, factor_of[p][1] * m)

    used = pd.DataFrame([dict(rec=rec, person=p, radar_entry=th.radar_threshold[rec], radar_exit_delta=radar_of[p][0],
                              radar_confirm_eps=radar_of[p][1], radar_exit=levels(rec, p, *radar_of[p])[0][0],
                              radar_confirm=levels(rec, p, *radar_of[p])[0][1], mocap_entry=th.mocap_threshold[rec],
                              mocap_exit_factor=factor_of[p][0], mocap_confirm_factor=factor_of[p][1],
                              mocap_exit=levels(rec, p, *radar_of[p])[1][0],
                              mocap_confirm=levels(rec, p, *radar_of[p])[1][1],
                              max_gap_s=rule_of[p].max_gap * EPOCH_S, min_bout_s=rule_of[p].min_bout * EPOCH_S)
                         for rec, p, *_ in res])
    used.to_csv(f'{out}/rules.csv', index=False)
    print('MoCap (exit, confirmation factor, max_gap_s, min_bout_s):',
          used.groupby(['mocap_exit_factor', 'mocap_confirm_factor', 'max_gap_s', 'min_bout_s']).size().to_dict(),
          '| radar (exit delta, confirmation eps):', used.groupby(['radar_exit_delta', 'radar_confirm_eps']).size()
          .to_dict())

    # evaluation with the calibrated definition
    ev_rows = [evaluate(rec, rs, ms, lab, rule_of[p], levels(rec, p, *radar_of[p])) for rec, p, rs, ms, lab, _ in res]
    events = pd.concat([r[0] for r in ev_rows], ignore_index=True)
    events.to_csv(f'{out}/bouts.csv', index=False)
    scores = pd.DataFrame([r[1] for r in ev_rows])
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
          f'{m.offset_err_s.mean():+.2f} s (radar - MoCap); onset and offset both within 1 s: {within:.2f}; offsets '
          f'late / early: {(m.offset_err_s > 0).sum()} / {(m.offset_err_s < 0).sum()}')
    print(f'lowest bout F1:\n{scores.nsmallest(8, "bout_f1").round(3).to_string(index=False)}')
    for src, g in events.groupby('source'):
        print(f'  {src} bout duration (s), all bouts: {spread(g.duration_s, ".1f")}, '
              f'matched {g.matched.mean():.2f}; unmatched bouts {spread(g[~g.matched].duration_s, ".1f")}')
    per_label = events.groupby(['label', 'source']).agg(bouts=('matched', 'size'),
                                                        unmatched=('matched', lambda x: 1 - x.mean())).unstack('source')
    print(f'\nper scripted activity (label at the middle of the bout), at least 60 MoCap bouts:\n'
          f'{per_label[per_label[("bouts", "mocap")] >= 60].round(2).to_string()}')

    mt = pd.concat([r[2] for r in ev_rows], ignore_index=True)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    agreement = agreement_table(mt)
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(f'\nper-recording metrics:\n{agreement.round(4).to_string(index=False)}')
    for k, g in mt.groupby('metric', sort=False):
        per_person = (g.radar - g.mocap).groupby(g.rec.str.split('/').str[0]).mean()
        print(f'  {k:20s} MoCap {spread(g.mocap)}  radar {spread(g.radar)}  mean error per person '
              f'{per_person.min():+.3f} ({per_person.idxmin()}) to {per_person.max():+.3f} ({per_person.idxmax()})')

    # sensitivity: the radar calibrated on bout F1 instead of the count (leaving the person out, from the same training
    # table), fixed radar thresholds, and neither hysteresis nor confirmation on either sensor
    f1_of = {p: tuple(g.loc[g.bout_f1.idxmax(), ['delta', 'eps']]) for p, g in radar_table.groupby('person')}
    variants = {'radar calibrated on bout F1': lambda rec, p: levels(rec, p, *f1_of[p])}
    for e in (0, 0.3, 0.5, 0.8):
        for d in (0.2, 0.3):
            variants[f'radar delta {d}, eps {e}'] = lambda rec, p, d=d, e=e: levels(rec, p, d, e)
    variants['no hysteresis, no confirmation'] = lambda rec, p: (
        (th.radar_threshold[rec],) * 2, (th.mocap_threshold[rec],) * 2)
    sens = []
    for name, ex in variants.items():
        vr = [evaluate(rec, rs, ms, lab, rule_of[p], ex(rec, p)) for rec, p, rs, ms, lab, _ in res]
        sc = pd.DataFrame([r[1] for r in vr])
        sens.append(agreement_table(pd.concat([r[2] for r in vr], ignore_index=True)).assign(
            variant=name, bout_f1=sc.bout_f1.median(), f1_bout_epochs=sc.f1_bout_epochs.median()))
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(f'{out}/sensitivity.csv', index=False)
    print(f'\nsensitivity (relative MAE; MAE / no-sensor MAE; F1 medians over recordings):\n'
          f'{sens.pivot_table(index="variant", columns="metric", values="rel_mae", sort=False).round(3)}\n'
          f'{sens.pivot_table(index="variant", columns="metric", values="vs_baseline", sort=False).round(2)}\n'
          f'{sens.groupby("variant", sort=False)[["bout_f1", "f1_bout_epochs"]].first().round(3)}')
