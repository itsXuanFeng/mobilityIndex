"""
Calibrate the MoCap bout definition (mobility/bouts.py: hysteresis down to exit_factor x the active threshold,
confirmation at confirm_factor x the active threshold or by a MoCap posture transition, fill gaps up to max_gap, drop
bouts shorter than min_bout) against the activity labels, on the MoCap joint speed of every recording with MoCap and
labels. No radar data is needed. Needs mocap_epochs.csv and mocap_active_threshold.csv of the latest mocap_threshold
run (scripts/calibrate_mocap_threshold.py) and the MoCap posture rules of the latest posture_rule run
(scripts/calibrate_posture.py). The posture transitions are found on the 50 Hz skeleton, under each fold's posture
rule.

For every person: the combination that best separates active from inactive label epochs of all *other* persons
(balanced accuracy), with MoCap active defined by that person's fold threshold and the transitions by that person's
fold posture rule, and how well it does on that person (F1 against the labels, before and after). 'ALL' (every
person) is for people outside the calibration.

    python scripts/calibrate_bout_rule.py --workers 16
    python scripts/calibrate_bout_rule.py --workers 16 --no-transitions --tag no-transitions   # as until 2026-11-24

Writes to its run folder, output/experiments/bout_rule_<time>/:
    bout_rule.csv        person, mocap_exit_factor, mocap_confirm_factor, max_gap_s, min_bout_s, training balanced
                         accuracy, held-out F1 before (no hysteresis, no confirmation, no rule) / after
    bout_rule_grid.csv   every candidate, all persons: label epochs in and out of bouts, balanced accuracy, F1
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import (BOUT_RULE_FILE, balanced_accuracy, calibrate_bout_rule, rule_confusion,  # noqa: E402
                            transition_epochs)
from mobility.cli import parallel_map  # noqa: E402
from mobility.config import EPOCH_S  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.groundtruth import active_speed_thresholds, mocap_50hz  # noqa: E402
from mobility.io.mocap import MOCAP_FPS  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.posture import posture_rules  # noqa: E402
from mobility.transitions import posture_transitions  # noqa: E402

KEYS = ['exit_factor', 'confirm_factor', 'max_gap', 'min_bout']
RULES = {}   # {fold: MoCap posture rule}, filled before the workers are forked


def transition_masks(job):
    """{fold: (E,) bool} for one recording: its epochs (start times t, mocap_epochs.csv) that contain a MoCap posture
    transition under each fold's posture rule, found on the 50 Hz skeleton"""
    (person, scenario), t = job
    sig = mocap_50hz(person, scenario)
    out = {}
    for fold, rule in RULES.items():
        ev = posture_transitions(sig['t'], rule.classify(sig), rule.min_hold_s, fps=MOCAP_FPS)
        out[fold] = transition_epochs(t, ev.t.values, EPOCH_S)
    return out


def f1(c):
    return 2 * c.tp / (2 * c.tp + c.fp + c.fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    ap.add_argument('--no-transitions', action='store_true',
                    help='runs are confirmed by the level only (the definition until 2026-11-24)')
    add_run_args(ap)
    args = ap.parse_args()
    with run('bout_rule', args, partial=False) as run_dir:
        ep = pd.read_csv(latest('mocap_threshold', 'mocap_epochs.csv'))
        thr = active_speed_thresholds()
        persons = sorted(ep.person.unique(), key=natural_key)
        print(f'{ep.groupby(["person", "scenario"]).ngroups} recordings, {len(persons)} persons', flush=True)
        transitions = None
        if not args.no_transitions:
            rules = posture_rules()
            RULES.update({p: rules[p] for p in persons + ['ALL']})
            jobs = [(k, g.sort_values('t').t.values) for k, g in ep.groupby(['person', 'scenario'], sort=False)]
            transitions = dict(zip([k for k, _ in jobs], parallel_map(transition_masks, jobs, args.workers)))
            n = sum(m['ALL'].sum() for m in transitions.values())
            print(f'MoCap posture transitions (ALL rule): {n} epochs with a transition', flush=True)
        conf = rule_confusion(ep, {p: thr[p] for p in persons + ['ALL']}, transitions)

        rows = []
        for person in persons + [None]:
            rule, f_exit, f_conf, ba = calibrate_bout_rule(conf, exclude_person=person)
            row = dict(person=person or 'ALL', mocap_exit_factor=f_exit, mocap_confirm_factor=f_conf,
                       max_gap_s=rule.max_gap * EPOCH_S, min_bout_s=rule.min_bout * EPOCH_S, train_balanced_accuracy=ba)
            if person:
                held = conf[(conf.fold == person) & (conf.person == person)]
                held = held.groupby(KEYS)[['tp', 'fn', 'tn', 'fp']].sum()
                row.update(f1_before=f1(held.loc[(1.0, 1.0, 0, 1)]),
                           f1_after=f1(held.loc[(f_exit, f_conf, rule.max_gap, rule.min_bout)]))
            rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(run_dir, BOUT_RULE_FILE), index=False)
        print(f'rule per left-out person:\n{out.round(4).to_string(index=False)}')
        held = out[out.person != 'ALL']
        defs = out.groupby(['mocap_exit_factor', 'mocap_confirm_factor', 'max_gap_s', 'min_bout_s']).size()
        print(f'(exit, confirmation, max_gap_s, min_bout_s): number of folds {defs.to_dict()}')
        print(f'\nheld-out F1 against the labels, median [min-max]: before {held.f1_before.median():.3f} '
              f'[{held.f1_before.min():.3f}-{held.f1_before.max():.3f}], after {held.f1_after.median():.3f} '
              f'[{held.f1_after.min():.3f}-{held.f1_after.max():.3f}]')

        a = conf[conf.fold == 'ALL']
        grid = a.groupby(KEYS)[['tp', 'fn', 'tn', 'fp']].sum()
        grid = grid.assign(balanced_accuracy=balanced_accuracy(grid), f1=f1(grid))
        grid.reset_index().to_csv(os.path.join(run_dir, 'bout_rule_grid.csv'), index=False)
        best = grid.balanced_accuracy.groupby(['confirm_factor', 'exit_factor']).max().unstack()
        print(f'\nbalanced accuracy, all persons (rows confirmation factor, columns exit factor; best max_gap, '
              f'min_bout):\n{best.round(4).to_string()}')
        print(f'written {run_dir}')
