"""
Category 2 (activity bout structure): radar vs MoCap on the recordings evaluated in Category 1. Bouts start from the
Category 1 active / inactive calls, so this reuses its leave-one-subject-out thresholds as the entry thresholds of the
hysteresis. Run first:
    scripts/evaluate_category1.py     (thresholds.csv: radar and MoCap active thresholds per recording)
    scripts/calibrate_bout_rule.py    (MoCap exit and confirmation factors and bout rule per left-out person)
The radar thresholds are calibrated here against the MoCap bouts, leaving the person out: the confirmation on the bout
count, the exit on the mean bout duration.

    python scripts/evaluate_category2.py --workers 16
    python scripts/evaluate_category2.py --only person_1/scenario_1 person_5/scenario_3

Writes to its run folder, output/experiments/category2_<time>/:
    rules.csv              per recording: entry, exit and confirmation thresholds of both sensors, the bout rule
    radar_levels.csv       per left-out person and radar (exit delta, confirmation eps): training count error,
                           mean-duration error, bout F1
    epoch_scores.csv       per recording: F1 of the radar's active calls and of its in-bout calls (MoCap as reference)
    bout_scores.csv        per recording: bouts of each sensor, matched, precision, recall, F1, onset / offset error
    bouts.csv              every bout of both sensors, matched or not, with the activity label at its middle
    metrics.csv            per recording: the Category 2 metrics, radar vs MoCap
    agreement.csv          per metric across recordings: mean absolute error, bias, no-sensor baseline
    sensitivity.csv        the same agreement and bout F1 for the radar exit calibrated on the count too, the radar
                           calibrated on bout F1, fixed radar thresholds, and without hysteresis and confirmation
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.bouts import bout_rules, mocap_level_factors  # noqa: E402
from mobility.cli import error_summary, parallel_map, wide_tables  # noqa: E402
from mobility.config import EPOCH_S  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.utils import fold_of, person_of, spread  # noqa: E402


def prepare(th):
    """one recording, in a worker: radar and MoCap States on the same valid epochs, and the label per epoch"""
    try:
        s = load_session(*parse_recording(th.rec))
        rs, ms = ev.paired_states(s, th.radar_threshold, th.mocap_threshold, active_feature=th.active_feature,
                                  energy_feature=th.energy_feature)
        return th.rec, th.person, rs, ms, s.label_per_epoch()[:len(rs.t)], None
    except Exception:
        return th.rec, th.person, None, None, None, f'{th.rec}: {error_summary()}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    add_run_args(ap)
    args = ap.parse_args()
    with run('category2', args, partial=bool(args.only)) as run_dir:
        th = pd.read_csv(latest('category1', 'thresholds.csv'))   # recordings evaluated in Category 1
        if args.only:
            th = th[th.rec.isin(args.only)]
        rows = [r for _, r in th.iterrows()]
        out = run_dir
        wide_tables(max_columns=20)

        res = parallel_map(prepare, rows, args.workers)
        failed = [r[-1] for r in res if r[-1]]
        if failed:
            print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
        res = [r for r in res if not r[-1]]
        th = th[th.rec.isin({r[0] for r in res})].set_index('rec')
        print(f'{len(res)} recordings, {th.person.nunique()} persons', flush=True)

        # definitions, all calibrated without the evaluated person
        rules, factors = bout_rules(), mocap_level_factors()
        rule_of = {p: fold_of(rules, p) for p in th.person.unique()}
        factor_of = {p: fold_of(factors, p) for p in th.person.unique()}
        radar_thr = th.groupby('person').radar_threshold.first().to_dict()
        mocap_thr = th.groupby('person').mocap_threshold.first().to_dict()
        states = [(p, rs, ms) for _, p, rs, ms, _, _ in res]
        radar_of, radar_table = ev.loso_radar_bout_levels(states, radar_thr, mocap_thr, factor_of, rule_of)
        radar_table.to_csv(f'{out}/radar_levels.csv', index=False)

        def levels(rec, p, delta, eps):
            """((radar exit, confirmation), (MoCap exit, confirmation)) thresholds of a recording of person p"""
            r, m = th.radar_threshold[rec], th.mocap_threshold[rec]
            return (r - delta, r + eps), (factor_of[p][0] * m, factor_of[p][1] * m)

        used = pd.DataFrame([dict(rec=rec, person=p, radar_entry=th.radar_threshold[rec],
                                  radar_exit_delta=radar_of[p][0], radar_confirm_eps=radar_of[p][1],
                                  radar_exit=levels(rec, p, *radar_of[p])[0][0],
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
        ev_rows = [ev.evaluate_bouts(rec, rs, ms, lab, rule_of[p], levels(rec, p, *radar_of[p]))
                   for rec, p, rs, ms, lab, _ in res]
        events = pd.concat([r[0] for r in ev_rows], ignore_index=True)
        events.to_csv(f'{out}/bouts.csv', index=False)
        scores = pd.DataFrame([r[1] for r in ev_rows])
        scores[['rec', 'epochs', 'f1_active', 'f1_bout_epochs']].to_csv(f'{out}/epoch_scores.csv', index=False)
        scores.drop(columns=['f1_active', 'f1_bout_epochs']).to_csv(f'{out}/bout_scores.csv', index=False)
        print(f'\nper epoch, F1 over recordings, median [10th-90th percentile]: active calls '
              f'{spread(scores.f1_active)}, in-bout calls {spread(scores.f1_bout_epochs)}')
        print(f'per bout (IoU >= {ev.BOUT_MATCH_IOU}): F1 {spread(scores.bout_f1)}, precision '
              f'{spread(scores.bout_precision)}, recall {spread(scores.bout_recall)}')
        print(f'  pooled: {scores.matched.sum()} matched of {scores.mocap_bouts.sum()} MoCap and '
              f'{scores.radar_bouts.sum()} radar bouts; matched bouts: onset error '
              f'{spread(scores.onset_mae_s, ".2f")} s, '
              f'offset error {spread(scores.offset_mae_s, ".2f")} s (per-recording MAE)')
        m = events.dropna(subset=['onset_err_s'])
        within = ((m.onset_err_s.abs() <= 1) & (m.offset_err_s.abs() <= 1)).mean()
        print(f'  pooled over matched bouts: mean onset error {m.onset_err_s.mean():+.2f} s, offset '
              f'{m.offset_err_s.mean():+.2f} s (radar - MoCap); onset and offset both within 1 s: {within:.2f}; '
              f'offsets '
              f'late / early: {(m.offset_err_s > 0).sum()} / {(m.offset_err_s < 0).sum()}')
        print(f'lowest bout F1:\n{scores.nsmallest(8, "bout_f1").round(3).to_string(index=False)}')
        for src, g in events.groupby('source'):
            print(f'  {src} bout duration (s), all bouts: {spread(g.duration_s, ".1f")}, '
                  f'matched {g.matched.mean():.2f}; unmatched bouts {spread(g[~g.matched].duration_s, ".1f")}')
        per_label = events.groupby(['label', 'source']).agg(
            bouts=('matched', 'size'), unmatched=('matched', lambda x: 1 - x.mean())).unstack('source')
        print(f'\nper scripted activity (label at the middle of the bout), at least 60 MoCap bouts:\n'
              f'{per_label[per_label[("bouts", "mocap")] >= 60].round(2).to_string()}')

        mt = pd.concat([r[2] for r in ev_rows], ignore_index=True)
        mt.to_csv(f'{out}/metrics.csv', index=False)
        agreement = ev.agreement_with_baseline(mt, ratio=True)
        agreement.to_csv(f'{out}/agreement.csv', index=False)
        print(f'\nper-recording metrics:\n{agreement.round(4).to_string(index=False)}')
        for k, g in mt.groupby('metric', sort=False):
            per_person = (g.radar - g.mocap).groupby(person_of(g.rec)).mean()
            print(f'  {k:20s} MoCap {spread(g.mocap)}  radar {spread(g.radar)}  mean error per person '
                  f'{per_person.min():+.3f} ({per_person.idxmin()}) to {per_person.max():+.3f} ({per_person.idxmax()})')

        # sensitivity, leaving the person out from the same training table: the radar exit calibrated on the count too
        # (both thresholds on the count), and the radar calibrated on bout F1; then fixed radar thresholds, and neither
        # hysteresis nor confirmation on either sensor
        count_of = {p: tuple(g.sort_values(['count_abs_err', 'eps', 'delta']).iloc[0][['delta', 'eps']])
                    for p, g in radar_table.groupby('person')}
        f1_of = {p: tuple(g.loc[g.bout_f1.idxmax(), ['delta', 'eps']]) for p, g in radar_table.groupby('person')}
        variants = {'radar exit and confirmation on the count': lambda rec, p: levels(rec, p, *count_of[p]),
                    'radar calibrated on bout F1': lambda rec, p: levels(rec, p, *f1_of[p])}
        for e in (0, 0.3, 0.5, 0.8):
            for d in (0.3, 0.5):
                variants[f'radar delta {d}, eps {e}'] = lambda rec, p, d=d, e=e: levels(rec, p, d, e)
        variants['no hysteresis, no confirmation'] = lambda rec, p: (
            (th.radar_threshold[rec],) * 2, (th.mocap_threshold[rec],) * 2)
        sens = []
        for name, ex in variants.items():
            vr = [ev.evaluate_bouts(rec, rs, ms, lab, rule_of[p], ex(rec, p)) for rec, p, rs, ms, lab, _ in res]
            sc = pd.DataFrame([r[1] for r in vr])
            sens.append(ev.agreement_with_baseline(pd.concat([r[2] for r in vr], ignore_index=True), ratio=True).assign(
                variant=name, bout_f1=sc.bout_f1.median(), f1_bout_epochs=sc.f1_bout_epochs.median()))
        sens = pd.concat(sens, ignore_index=True)
        sens.to_csv(f'{out}/sensitivity.csv', index=False)
        print(f'\nsensitivity (relative MAE; MAE / no-sensor MAE; F1 medians over recordings):\n'
              f'{sens.pivot_table(index="variant", columns="metric", values="rel_mae", sort=False).round(3)}\n'
              f'{sens.pivot_table(index="variant", columns="metric", values="vs_baseline", sort=False).round(2)}\n'
              f'{sens.groupby("variant", sort=False)[["bout_f1", "f1_bout_epochs"]].first().round(3)}')
