"""
Category 2 (activity bout structure): radar vs MoCap on the recordings evaluated in Category 1. Bouts start from the
Category 1 active / inactive calls, so this reuses its leave-one-subject-out thresholds as the entry thresholds of the
hysteresis. Run first:
    scripts/evaluate_category1.py     (thresholds.csv: radar and MoCap active thresholds per recording)
    scripts/calibrate_bout_rule.py    (MoCap exit and confirmation factors and bout rule per left-out person)
The radar bout definition is calibrated here against the MoCap bouts, leaving the person out: one margin kappa on the
radar's log10 level, the same below the entry (exit) and above it (confirmation), chosen on the bout count. With
--confirm-speed, a bout must also reach a Doppler speed, calibrated together with kappa (an option, not the default:
a fixed speed may miss the bouts of people who move more slowly).

    python scripts/evaluate_category2.py --workers 16
    python scripts/evaluate_category2.py --workers 16 --confirm-speed --tag speed
    python scripts/evaluate_category2.py --only person_1/scenario_1 person_5/scenario_3 --tag test

Writes to its run folder, output/experiments/category2[-<tag>]_<time>/:
    rules.csv              per recording: entry, exit and confirmation thresholds of both sensors, the radar margin
                           kappa (also as radar_exit_delta = radar_confirm_eps, as read by the personal calibration),
                           the radar confirmation speed (empty without), the bout rule
    radar_levels.csv       per left-out person, kappa and confirmation speed: training count error, mean-duration
                           error, bout F1 (both the level-only and the speed grid)
    epoch_scores.csv       per recording: F1 of the radar's active calls and of its in-bout calls (MoCap as reference),
                           and of each sensor's in-bout calls against the activity labels
    bout_scores.csv        per recording: bouts of each sensor, matched, precision, recall, F1, onset / offset error
    bouts.csv              every bout of both sensors, matched or not, with the activity label at its middle
    metrics.csv            per recording: the Category 2 metrics, radar vs MoCap
    agreement.csv          per metric across recordings: mean absolute error, bias, no-sensor baseline
    sensitivity.csv        the same agreement, bout F1 and label F1 for other radar definitions: the earlier untied
                           exit / confirmation, kappa calibrated on bout F1, with / without the speed confirmation,
                           the speed confirmation with the radar speeds scaled down (slower movement), fixed kappa,
                           and neither hysteresis nor confirmation on either sensor
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
from mobility.labels import time_classes  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.utils import fold_of, person_of, spread  # noqa: E402

SPEED_SCALES = (0.8, 0.7)   # stress test: the radar speeds scaled down, the calibrated thresholds kept
F1_COLUMNS = ['bout_f1', 'f1_bout_epochs', 'f1_labels_radar']


def prepare(th):
    """one recording, in a worker: radar and MoCap States on the same valid epochs, the label and the label class per
    epoch (class at the middle of the epoch, as the MoCap calibration against the labels)"""
    try:
        s = load_session(*parse_recording(th.rec))
        rs, ms = ev.paired_states(s, th.radar_threshold, th.mocap_threshold, active_feature=th.active_feature,
                                  energy_feature=th.energy_feature)
        cls = time_classes(rs.t + EPOCH_S / 2, s.labels)
        return th.rec, th.person, rs, ms, s.label_per_epoch()[:len(rs.t)], cls, None
    except Exception:
        return th.rec, th.person, None, None, None, None, f'{th.rec}: {error_summary()}'


def speed_label(c):
    return 'none' if c is None else f'{c:g} m/s'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    ap.add_argument('--confirm-speed', action='store_true',
                    help='the radar bouts must also reach a Doppler speed (calibrated); default: level only')
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
        res = [r[:-1] for r in res if not r[-1]]
        th = th[th.rec.isin({r[0] for r in res})].set_index('rec')
        print(f'{len(res)} recordings, {th.person.nunique()} persons', flush=True)

        # definitions, all calibrated without the evaluated person
        rules, factors = bout_rules(), mocap_level_factors()
        rule_of = {p: fold_of(rules, p) for p in th.person.unique()}
        factor_of = {p: fold_of(factors, p) for p in th.person.unique()}
        radar_thr = th.groupby('person').radar_threshold.first().to_dict()
        mocap_thr = th.groupby('person').mocap_threshold.first().to_dict()
        states = [(p, rs, ms) for _, p, rs, ms, _, _ in res]
        cal = (states, radar_thr, mocap_thr, factor_of, rule_of)
        level_of, level_table = ev.loso_radar_bout_levels(*cal)                                   # kappa only
        speed_of, speed_table = ev.loso_radar_bout_levels(*cal, speeds=ev.RADAR_CONFIRM_SPEEDS)   # kappa + speed
        untied_of, _ = ev.loso_radar_untied_levels(*cal)                                          # earlier definition
        pd.concat([level_table, speed_table], ignore_index=True).to_csv(f'{out}/radar_levels.csv', index=False)
        radar_of = speed_of if args.confirm_speed else level_of
        print(f'radar definition: {"kappa + confirmation speed" if args.confirm_speed else "kappa, level only"}')

        def levels(rec, p, kappa, speed=None):
            """((radar exit, confirmation), (MoCap exit, confirmation), radar confirmation speed) of a recording of
            person p, with radar margin kappa"""
            r, m = th.radar_threshold[rec], th.mocap_threshold[rec]
            return (r - kappa, r + kappa), (factor_of[p][0] * m, factor_of[p][1] * m), speed

        def untied_levels(rec, p, delta, eps):
            r, m = th.radar_threshold[rec], th.mocap_threshold[rec]
            return (r - delta, r + eps), (factor_of[p][0] * m, factor_of[p][1] * m), None

        used = []
        for rec, p, *_ in res:
            k, c = radar_of[p]
            (r_exit, r_conf), (m_exit, m_conf), _ = levels(rec, p, k, c)
            used.append(dict(rec=rec, person=p, radar_entry=th.radar_threshold[rec], radar_kappa=k,
                             radar_exit_delta=k, radar_confirm_eps=k, radar_exit=r_exit, radar_confirm=r_conf,
                             radar_confirm_speed=c, mocap_entry=th.mocap_threshold[rec],
                             mocap_exit_factor=factor_of[p][0], mocap_confirm_factor=factor_of[p][1],
                             mocap_exit=m_exit, mocap_confirm=m_conf,
                             max_gap_s=rule_of[p].max_gap * EPOCH_S, min_bout_s=rule_of[p].min_bout * EPOCH_S))
        used = pd.DataFrame(used)
        used.to_csv(f'{out}/rules.csv', index=False)
        print('MoCap (exit, confirmation factor, max_gap_s, min_bout_s):',
              used.groupby(['mocap_exit_factor', 'mocap_confirm_factor', 'max_gap_s', 'min_bout_s']).size().to_dict())
        for name, chosen in (('kappa, level only', level_of), ('kappa + speed', speed_of),
                             ('untied (delta, eps)', untied_of)):
            n = pd.Series({rec: chosen[p] for rec, p, *_ in res}).value_counts()
            print(f'radar {name}, recordings per choice: {n.to_dict()}')

        # evaluation with the calibrated definition
        ev_rows = [ev.evaluate_bouts(rec, rs, ms, lab, rule_of[p], levels(rec, p, *radar_of[p]), cls)
                   for rec, p, rs, ms, lab, cls in res]
        events = pd.concat([r[0] for r in ev_rows], ignore_index=True)
        events.to_csv(f'{out}/bouts.csv', index=False)
        scores = pd.DataFrame([r[1] for r in ev_rows])
        epoch_cols = ['rec', 'epochs', 'f1_active', 'f1_bout_epochs', 'f1_labels_radar', 'f1_labels_mocap']
        scores[epoch_cols].to_csv(f'{out}/epoch_scores.csv', index=False)
        scores.drop(columns=epoch_cols[2:]).to_csv(f'{out}/bout_scores.csv', index=False)
        print(f'\nper epoch, F1 over recordings, median [10th-90th percentile]: active calls '
              f'{spread(scores.f1_active)}, in-bout calls {spread(scores.f1_bout_epochs)}')
        print(f'in-bout calls against the activity labels: radar {spread(scores.f1_labels_radar)}, MoCap '
              f'{spread(scores.f1_labels_mocap)}')
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
              f'offsets late / early: {(m.offset_err_s > 0).sum()} / {(m.offset_err_s < 0).sum()}')
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

        # sensitivity: other radar definitions, each leaving the person out where calibrated; the MoCap definition as
        # calibrated, except in the last row
        table = speed_table if args.confirm_speed else level_table
        best_f1 = table.loc[table.groupby('person').bout_f1.idxmax()].set_index('person')
        f1_of = {p: (r.kappa, None if pd.isna(r.confirm_speed) else r.confirm_speed) for p, r in best_f1.iterrows()}
        variants = {  # name: (levels(rec, p), radar speed scale)
            'untied: exit on duration, confirmation on count (until 2026-11-21)':
                (lambda rec, p: untied_levels(rec, p, *untied_of[p]), 1.0),
            'kappa calibrated on bout F1': (lambda rec, p: levels(rec, p, *f1_of[p]), 1.0),
            'kappa, level only': (lambda rec, p: levels(rec, p, *level_of[p]), 1.0),
            'kappa + confirmation speed': (lambda rec, p: levels(rec, p, *speed_of[p]), 1.0)}
        for s in SPEED_SCALES:
            variants[f'kappa + confirmation speed, radar speeds x{s:g}'] = (
                lambda rec, p: levels(rec, p, *speed_of[p]), s)
        for k in (0.3, 0.5, 0.8):
            variants[f'fixed kappa {k:g}, level only'] = (lambda rec, p, k=k: levels(rec, p, k), 1.0)
        variants['no hysteresis, no confirmation (either sensor)'] = (lambda rec, p: (
            (th.radar_threshold[rec],) * 2, (th.mocap_threshold[rec],) * 2, None), 1.0)
        sens = []
        for name, (ex, scale) in variants.items():
            vr = []
            for rec, p, rs, ms, lab, cls in res:
                speed = rs.extra.get('speed')
                if scale != 1.0:
                    rs.extra['speed'] = speed * scale
                vr.append(ev.evaluate_bouts(rec, rs, ms, lab, rule_of[p], ex(rec, p), cls))
                rs.extra['speed'] = speed
            sc = pd.DataFrame([r[1] for r in vr])
            sens.append(ev.agreement_with_baseline(pd.concat([r[2] for r in vr], ignore_index=True), ratio=True).assign(
                variant=name, bout_f1=sc.bout_f1.median(), f1_bout_epochs=sc.f1_bout_epochs.median(),
                f1_labels_radar=sc.f1_labels_radar.median()))
        sens = pd.concat(sens, ignore_index=True)
        sens['bias_pct'] = 100 * sens.bias * sens.rel_mae / sens.mean_abs_err   # bias in % of the mean MoCap value
        sens.to_csv(f'{out}/sensitivity.csv', index=False)
        print(f'\nsensitivity (relative MAE; bias in % of the mean MoCap value; MAE / no-sensor MAE; F1 medians over '
              f'recordings):\n'
              f'{sens.pivot_table(index="variant", columns="metric", values="rel_mae", sort=False).round(3)}\n'
              f'{sens.pivot_table(index="variant", columns="metric", values="bias_pct", sort=False).round(1)}\n'
              f'{sens.pivot_table(index="variant", columns="metric", values="vs_baseline", sort=False).round(2)}\n'
              f'{sens.groupby("variant", sort=False)[F1_COLUMNS].first().round(3)}')
