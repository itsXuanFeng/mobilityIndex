"""
Calibrate the MoCap active threshold (RMS joint speed per 1 s epoch) against the activity labels (mobility/labels.py),
on every recording with MoCap and labels. No radar data is needed.

For every person: the threshold that best separates active from inactive label epochs of all *other* persons (balanced
accuracy), and how well it does on that person (AUC, F1). The evaluation uses the threshold calibrated without the
evaluated person; 'ALL' (every person) is for people outside the calibration.

    python scripts/calibrate_mocap_threshold.py [--workers 8]

Writes OUTPUT_ROOT/groundtruth/:
    mocap_active_threshold.csv   person, threshold, labelled epochs, held-out AUC and F1
    mocap_epochs.csv             every epoch: person, scenario, time, RMS joint speed, label class
    per_label.csv                per label: median speed, share above the 'ALL' threshold
"""
import argparse
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import EPOCH_S, GROUNDTRUTH_DIR, LABEL_ROOT  # noqa: E402
from mobility.groundtruth import THRESHOLD_FILE, calibrate_active_speed, mocap_epochs  # noqa: E402
from mobility.io.recordings import find_recordings, labels, natural_key  # noqa: E402
from mobility.labels import ACTIVE, INACTIVE, label_class  # noqa: E402


def epochs_of(rec):
    try:
        return mocap_epochs(*rec)
    except Exception:
        traceback.print_exc()
        print(f'FAILED {"/".join(rec)}', flush=True)
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    recs = [r for r in find_recordings() if os.path.exists(os.path.join(LABEL_ROOT, *r, 'timestamps.csv'))]
    print(f'{len(recs)} recordings with labels', flush=True)
    with Pool(args.workers) as pool:
        parts = pool.map(epochs_of, recs)
    ep = pd.concat([p for p in parts if p is not None], ignore_index=True)
    print(f'{ep.groupby(["person", "scenario"]).ngroups} recordings with MoCap, {len(ep)} epochs '
          f'({(ep.cls == ACTIVE).sum()} active, {(ep.cls == INACTIVE).sum()} inactive by label)')
    os.makedirs(GROUNDTRUTH_DIR, exist_ok=True)
    ep.to_csv(os.path.join(GROUNDTRUTH_DIR, 'mocap_epochs.csv'), index=False)

    lab = ep[ep.cls.isin([ACTIVE, INACTIVE])]
    rows = []
    for person in sorted(ep.person.unique(), key=natural_key):
        thr = calibrate_active_speed(ep, exclude_person=person)
        held = lab[lab.person == person]
        y = (held.cls == ACTIVE).values
        rows.append(dict(person=person, threshold=thr, epochs=len(held), auc=roc_auc_score(y, held.speed),
                         f1=f1_score(y, held.speed > thr)))
    all_thr = calibrate_active_speed(ep)
    rows.append(dict(person='ALL', threshold=all_thr, epochs=len(lab), auc=np.nan, f1=np.nan))
    th = pd.DataFrame(rows)
    th.to_csv(THRESHOLD_FILE, index=False)
    per = th[th.person != 'ALL']
    print(f'\nthreshold without each person: {per.threshold.min():.3f}-{per.threshold.max():.3f} m/s '
          f'(all persons {all_thr:.3f}); held-out AUC median {per.auc.median():.3f} '
          f'[{per.auc.min():.3f}-{per.auc.max():.3f}], F1 median {per.f1.median():.3f} '
          f'[{per.f1.min():.3f}-{per.f1.max():.3f}]')
    print(th.round(3).to_string(index=False))

    # per label, on the MoCap clock
    rows = []
    for (person, scenario), g in ep.groupby(['person', 'scenario']):
        lb = labels(person, scenario)
        mid = g.t.values + EPOCH_S / 2
        for _, r in lb.iterrows():
            m = (mid >= r.start) & (mid <= r.end)
            rows += [dict(label=r.label, speed=s) for s in g.speed.values[m]]
    pl = pd.DataFrame(rows)
    summary = pl.groupby('label').agg(epochs=('speed', 'size'), median_speed=('speed', 'median'),
                                      share_active=('speed', lambda s: (s > all_thr).mean()))
    summary['class'] = [label_class(lab_) for lab_ in summary.index]
    summary = summary.sort_values('median_speed')
    summary.to_csv(os.path.join(GROUNDTRUTH_DIR, 'per_label.csv'))
    pd.set_option('display.width', 200)
    print('\n' + summary.round(3).to_string())
