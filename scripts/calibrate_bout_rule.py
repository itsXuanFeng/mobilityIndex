"""
Calibrate the MoCap bout definition (mobility/bouts.py: hysteresis down to exit_factor x the active threshold,
confirmation at confirm_factor x the active threshold, fill gaps up to max_gap, drop bouts shorter than min_bout)
against the activity labels, on the MoCap joint speed of every
recording with MoCap and labels. No radar data is needed.
Needs mocap_epochs.csv and mocap_active_threshold.csv of the latest mocap_threshold run: run
scripts/calibrate_mocap_threshold.py first.

For every person: the combination that best separates active from inactive label epochs of all *other* persons
(balanced accuracy), with MoCap active defined by that person's fold threshold, and how well it does on that person
(F1 against the labels, before and after). 'ALL' (every person) is for people outside the calibration.

    python scripts/calibrate_bout_rule.py

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
from mobility.bouts import BOUT_RULE_FILE, balanced_accuracy, calibrate_bout_rule, rule_confusion  # noqa: E402
from mobility.config import EPOCH_S  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.groundtruth import active_speed_thresholds  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402

KEYS = ['exit_factor', 'confirm_factor', 'max_gap', 'min_bout']


def f1(c):
    return 2 * c.tp / (2 * c.tp + c.fp + c.fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    args = ap.parse_args()
    with run('bout_rule', args, partial=False) as run_dir:
        ep = pd.read_csv(latest('mocap_threshold', 'mocap_epochs.csv'))
        thr = active_speed_thresholds()
        persons = sorted(ep.person.unique(), key=natural_key)
        print(f'{ep.groupby(["person", "scenario"]).ngroups} recordings, {len(persons)} persons', flush=True)
        conf = rule_confusion(ep, {p: thr[p] for p in persons + ['ALL']})

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
