"""
Calibrate the MoCap posture rule (mobility/posture.py) against the script, on every recording with MoCap and labels.
No radar data is needed.

For every person, on all *other* persons:
    hip_stand, trunk_lie    pelvis height (standing) and trunk angle (lying): each maximises the balanced accuracy of
                            its split on the 1 s epochs whose posture the script fixes;
    low_floor, min_hold_s   the floor threshold (lowest of pelvis and knees) and minimum posture duration whose MoCap
                            transitions best reproduce the scripted postural changes (F1, same kind, within the label
                            +- 1 s), on the grid GRID_LOW_FLOOR x GRID_MIN_HOLD_S.
Then how the rule does on that person: F1 per posture against the script, and how many scripted changes its MoCap
transitions reproduce. 'ALL' (every person) is for people outside the calibration.

    python scripts/calibrate_posture.py [--workers 16]

Writes OUTPUT_ROOT/groundtruth/:
    posture_rule.csv       person, thresholds, min_hold_s, training F1, held-out posture F1 and transition F1
    posture_epochs.csv     every epoch: person, scenario, time, pelvis, knee, trunk, scripted posture
    posture_scripted.csv   per scripted change, held out: label, kind, whether MoCap reproduces it
    posture_extra.csv      MoCap transitions inside the script that match no scripted change, held out: kind, label
"""
import argparse
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import GROUNDTRUTH_DIR, LABEL_ROOT  # noqa: E402
from mobility.groundtruth import mocap_50hz  # noqa: E402
from mobility.io.mocap import MOCAP_FPS  # noqa: E402
from mobility.io.recordings import find_recordings, labels, natural_key  # noqa: E402
from mobility.labels import caregiver_period, script_period  # noqa: E402
from mobility.posture import (GRID_LOW_FLOOR, GRID_MIN_HOLD_S, POSTURE_RULE_FILE, POSTURES, UNKNOWN,  # noqa: E402
                              PostureRule, calibrate_thresholds, posture_epochs)
from mobility.transitions import match_scripted, posture_transitions, scripted_transitions  # noqa: E402


def load(rec):
    """epochs, 50 Hz posture signals and scripted changes of one recording"""
    try:
        ep = posture_epochs(*rec)
        if ep is None:
            return None
        sig = mocap_50hz(*rec)
        lb = labels(*rec)
        return ep, {k: sig[k] for k in ('t', 'hip_y', 'knee_y', 'trunk_deg')}, lb
    except Exception:
        traceback.print_exc()
        print(f'FAILED {"/".join(rec)}', flush=True)
        return None


def events_in_script(sig, lb, rule):
    """MoCap transitions inside the scripted period, outside the caregiver visit"""
    ev = posture_transitions(sig['t'], rule.classify(sig), rule.min_hold_s, fps=MOCAP_FPS)
    s0, s1 = script_period(lb)
    keep = (ev.t >= s0) & (ev.t <= s1)
    care = caregiver_period(lb)
    if care:
        keep &= ~((ev.t >= care[0]) & (ev.t <= care[1]))
    return ev[keep].reset_index(drop=True)


DATA = {}   # recording -> (epochs, 50 Hz signals, labels); filled before the worker pool forks


def transition_counts(rule):
    """per recording: matched scripted changes, MoCap transitions, scripted changes"""
    out = {}
    for rec, (_, sig, lb) in DATA.items():
        ev, sc = events_in_script(sig, lb, rule), scripted_transitions(lb)
        out[rec] = (len(match_scripted(sc, ev)), len(ev), len(sc))
    return out


def f1(counts):
    tp, ne, ns = np.sum(list(counts), axis=0) if counts else (0, 0, 0)
    return 2 * tp / (ne + ns) if ne + ns else np.nan


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    recs = [r for r in find_recordings() if os.path.exists(os.path.join(LABEL_ROOT, *r, 'timestamps.csv'))]
    with Pool(args.workers) as pool:
        parts = pool.map(load, recs)
    DATA.update({r: p for r, p in zip(recs, parts) if p is not None})
    ep = pd.concat([p[0] for p in DATA.values()], ignore_index=True)
    os.makedirs(GROUNDTRUTH_DIR, exist_ok=True)
    ep.to_csv(os.path.join(GROUNDTRUTH_DIR, 'posture_epochs.csv'), index=False)
    persons = sorted(ep.person.unique(), key=natural_key)
    known = ep[ep.posture != UNKNOWN]
    print(f'{len(DATA)} recordings with MoCap, {len(persons)} persons; scripted-posture epochs: '
          + ', '.join(f'{POSTURES[k]} {n}' for k, n in known.posture.value_counts().sort_index().items()), flush=True)

    # every candidate rule of every fold; the transition counts per recording are computed once per rule
    folds = {p: calibrate_thresholds(ep, exclude_person=p) for p in persons + [None]}
    grid = {p: [PostureRule(hip, low, trunk, hold) for low in GRID_LOW_FLOOR for hold in GRID_MIN_HOLD_S]
            for p, (hip, trunk) in folds.items()}
    todo = sorted({r for rules in grid.values() for r in rules}, key=str)
    with Pool(args.workers) as pool:
        counts = dict(zip(todo, pool.map(transition_counts, todo)))
    print(f'{len(todo)} candidate rules', flush=True)

    rows, scripted_rows, extra_rows = [], [], []
    for person in persons + [None]:
        train = {r: f1([c for rec, c in counts[r].items() if rec[0] != person]) for r in grid[person]}
        best = max(train.values())
        rule = min((r for r, v in train.items() if v == best), key=lambda r: (r.min_hold_s, r.low_floor))
        row = dict(person=person or 'ALL', hip_stand=rule.hip_stand, low_floor=rule.low_floor,
                   trunk_lie=rule.trunk_lie, min_hold_s=rule.min_hold_s, train_transition_f1=best)
        if person:
            held = known[known.person == person]
            pred = rule.classify(held)
            row.update(posture_f1=f1_score(held.posture, pred, labels=[0, 1, 2, 3], average='macro'),
                       **{f'f1_{POSTURES[k]}': f1_score(held.posture == k, pred == k) for k in range(4)},
                       transition_f1=f1([c for rec, c in counts[rule].items() if rec[0] == person]))
            for rec, (_, sig, lb) in DATA.items():
                if rec[0] != person:
                    continue
                ev, sc = events_in_script(sig, lb, rule), scripted_transitions(lb)
                m = match_scripted(sc, ev)
                scripted_rows += [dict(rec='/'.join(rec), label=s.label, kind=s.kind, found=i in m)
                                  for i, s in enumerate(sc.itertuples())]
                hit = set(m.values())
                for j, e in enumerate(ev.itertuples()):
                    if j not in hit:
                        inside = lb[(lb.start - 1 <= e.t) & (lb.end + 1 >= e.t)]
                        extra_rows.append(dict(rec='/'.join(rec), kind=e.kind,
                                               label=inside.label.iloc[0] if len(inside) else '(between labels)'))
        rows.append(row)
        print(f'{row["person"]:10s} {rule}  train F1 {best:.3f}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(POSTURE_RULE_FILE, index=False)
    held = out[out.person != 'ALL']
    print(f'\nthresholds without each person: pelvis {held.hip_stand.min():.3f}-{held.hip_stand.max():.3f} m, '
          f'floor {held.low_floor.min():.3f}-{held.low_floor.max():.3f} m, trunk {held.trunk_lie.min():.1f}-'
          f'{held.trunk_lie.max():.1f} deg; min hold {sorted(held.min_hold_s.unique())} s')
    print(f'held out, median [min-max]: posture macro F1 {held.posture_f1.median():.3f} '
          f'[{held.posture_f1.min():.3f}-{held.posture_f1.max():.3f}], transition F1 vs script '
          f'{held.transition_f1.median():.3f} [{held.transition_f1.min():.3f}-{held.transition_f1.max():.3f}]')
    print('per posture F1 (median over persons): ' + ', '.join(f'{p} {held[f"f1_{p}"].median():.3f}' for p in POSTURES))
    table = pd.DataFrame([dict(low_floor=r.low_floor, min_hold_s=r.min_hold_s, f1=f1(list(counts[r].values())))
                          for r in grid[None]])
    print(f'transition F1 vs script, all persons (rows floor threshold, columns min hold):\n'
          f'{table.pivot(index="low_floor", columns="min_hold_s", values="f1").round(3).to_string()}')

    sc = pd.DataFrame(scripted_rows)
    sc.to_csv(os.path.join(GROUNDTRUTH_DIR, 'posture_scripted.csv'), index=False)
    print(f'\nscripted changes reproduced by MoCap (held out), per kind:\n'
          f'{sc.groupby("kind").found.agg(["size", "sum", "mean"]).round(3).to_string()}')
    ex = pd.DataFrame(extra_rows)
    ex.to_csv(os.path.join(GROUNDTRUTH_DIR, 'posture_extra.csv'), index=False)
    print(f'\nMoCap transitions matching no scripted change: {len(ex)}, by label:\n'
          f'{ex.groupby(["label", "kind"]).size().sort_values(ascending=False).head(20).to_string()}')
    print(f'written {POSTURE_RULE_FILE}')
