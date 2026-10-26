"""
Calibrate the bout rule (mobility/bouts.py: fill gaps up to max_gap, drop bouts shorter than min_bout) against the
activity labels, on the MoCap active sequence of every recording with MoCap and labels. No radar data is needed.
Needs OUTPUT_ROOT/groundtruth/mocap_epochs.csv and mocap_active_threshold.csv: run scripts/calibrate_mocap_threshold.py
first.

For every person: the rule that best separates active from inactive label epochs of all *other* persons (balanced
accuracy), with MoCap active defined by that person's fold threshold, and how well it does on that person (F1 against
the labels, before and after the rule). 'ALL' (every person) is for people outside the calibration.

    python scripts/calibrate_bout_rule.py

Writes OUTPUT_ROOT/groundtruth/:
    bout_rule.csv        person, max_gap_s, min_bout_s, training balanced accuracy, held-out F1 before / after
    bout_rule_grid.csv   every candidate rule, all persons: label epochs in and out of bouts, balanced accuracy, F1
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import BOUT_RULE_FILE, balanced_accuracy, calibrate_bout_rule, rule_confusion  # noqa: E402
from mobility.config import EPOCH_S, GROUNDTRUTH_DIR  # noqa: E402
from mobility.groundtruth import active_speed_thresholds  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402


def f1(c):
    return 2 * c.tp / (2 * c.tp + c.fp + c.fn)


if __name__ == '__main__':
    ep = pd.read_csv(os.path.join(GROUNDTRUTH_DIR, 'mocap_epochs.csv'))
    thr = active_speed_thresholds()
    persons = sorted(ep.person.unique(), key=natural_key)
    print(f'{ep.groupby(["person", "scenario"]).ngroups} recordings, {len(persons)} persons', flush=True)
    conf = rule_confusion(ep, {p: thr[p] for p in persons + ['ALL']})

    rows = []
    for person in persons + [None]:
        rule, ba = calibrate_bout_rule(conf, exclude_person=person)
        row = dict(person=person or 'ALL', max_gap_s=rule.max_gap * EPOCH_S, min_bout_s=rule.min_bout * EPOCH_S,
                   train_balanced_accuracy=ba)
        if person:
            held = conf[(conf.fold == person) & (conf.person == person)].groupby(['max_gap', 'min_bout'])[
                ['tp', 'fn', 'tn', 'fp']].sum()
            row.update(f1_before=f1(held.loc[(0, 1)]), f1_after=f1(held.loc[(rule.max_gap, rule.min_bout)]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(BOUT_RULE_FILE, index=False)
    print(f'rule per left-out person:\n{out.round(4).to_string(index=False)}')
    held = out[out.person != 'ALL']
    print(f'\nheld-out F1 against the labels, median [min-max]: before the rule {held.f1_before.median():.3f} '
          f'[{held.f1_before.min():.3f}-{held.f1_before.max():.3f}], after {held.f1_after.median():.3f} '
          f'[{held.f1_after.min():.3f}-{held.f1_after.max():.3f}]')

    a = conf[conf.fold == 'ALL']
    grid = a.groupby(['max_gap', 'min_bout'])[['tp', 'fn', 'tn', 'fp']].sum()
    grid = grid.assign(balanced_accuracy=balanced_accuracy(grid), f1=f1(grid))
    grid.reset_index().to_csv(os.path.join(GROUNDTRUTH_DIR, 'bout_rule_grid.csv'), index=False)
    print(f'\nbalanced accuracy over the grid (all persons; rows max_gap, columns min_bout, epochs):\n'
          f'{grid.balanced_accuracy.unstack().round(3).to_string()}')
    print(f'written {BOUT_RULE_FILE}')
