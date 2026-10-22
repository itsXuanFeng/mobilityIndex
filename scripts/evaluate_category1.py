"""
Category 1 (total movement volume): radar vs MoCap on the recordings that have a cube.

    python scripts/evaluate_category1.py                       # every recording with a cube
    python scripts/evaluate_category1.py --only person_1/scenario_1 person_5/scenario_3

Writes to OUTPUT_ROOT/category1/:
    alignment.csv      time offset and its correlation per recording
    epoch_scores.csv   per feature and recording: AUC, F1 (thresholds leave-one-recording-out)
    metrics.csv        per recording: the Category 1 metrics, radar vs MoCap
    agreement.csv      per metric across recordings: mean absolute error and bias (times), Spearman (energy)
    per_label.csv      per scripted activity: MoCap vs radar active share
    distance.csv       distance dependence of each feature
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.config import OUTPUT_ROOT  # noqa: E402
from mobility.io.mocap import shot_for  # noqa: E402
from mobility.io.recordings import find_recordings, parse_recording  # noqa: E402
from mobility.radar.cube import cube_path  # noqa: E402
from mobility.session import load_session  # noqa: E402

FEATURES = ['n_cells', 'p_r4', 'p_sub', 'v_rms']
ACTIVE_FEATURE = 'n_cells'     # decides active / inactive
ENERGY_FEATURE = 'p_r4'        # total activity energy

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--overwrite', action='store_true', help='recompute cached features, MoCap signals, alignment')
    args = ap.parse_args()
    recs = [parse_recording(r) for r in args.only] if args.only else find_recordings()
    recs = [r for r in recs if os.path.exists(cube_path(*r))]
    no_mocap = [r for r in recs if shot_for(*r) is None]   # person_100-102: no MoCap, no labels
    if no_mocap:
        print('skipped, no MoCap:', ' '.join('/'.join(r) for r in no_mocap))
    recs = [r for r in recs if r not in no_mocap]
    out = os.path.join(OUTPUT_ROOT, 'category1')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    sessions = [load_session(p, s, args.overwrite) for p, s in recs]
    print(f'{len(sessions)} recordings')
    al = pd.DataFrame([dict(rec=s.name, shot=s.alignment['shot'], track=os.path.basename(s.alignment['track']),
                            offset_s=s.alignment['offset_s'], corr=s.alignment['corr'], margin=s.alignment['margin'])
                       for s in sessions])
    al.to_csv(f'{out}/alignment.csv', index=False)
    print(al.round(2).to_string(index=False), '\n')

    table = ev.epoch_table(sessions, FEATURES)
    n_ep = table.groupby('rec').agg(with_mocap=('ke', 'size'), in_window=('window', 'sum'),
                                    out_of_coverage=('covered', lambda c: (~c).sum()),
                                    caregiver=('no_caregiver', lambda c: (~c).sum()))
    print(f'epochs per recording\n{n_ep.to_string()}\n')

    thresholds, scores, distance = {}, [], []
    for f in FEATURES:
        thresholds[f] = ev.loro_thresholds(table, f)
        scores.append(ev.epoch_scores(table, f, thresholds[f]))
        distance.append(ev.distance_residuals(table, f).reset_index().rename(columns={'index': 'dist_m'})
                        .assign(feature=f))
    scores = pd.concat(scores)
    scores.to_csv(f'{out}/epoch_scores.csv', index=False)
    print('per-epoch scores, median over recordings:')
    print(scores.groupby('feature')[['auc', 'f1']].median().round(3).to_string(), '\n')
    for score in ('auc', 'f1'):
        print(f'{score} per recording:')
        print(scores.pivot(index='rec', columns='feature', values=score).round(2).to_string(), '\n')

    mt = ev.metric_table(sessions, thresholds[ACTIVE_FEATURE], category=1, active_feature=ACTIVE_FEATURE,
                         energy_feature=ENERGY_FEATURE)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    print(mt.pivot(index='rec', columns='metric', values=['radar', 'mocap']).round(2).to_string(), '\n')
    agreement = ev.metric_agreement(mt)
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(agreement.round(2).to_string(index=False), '\n')

    pooled = ev.best_threshold(table.loc[table.window, ACTIVE_FEATURE].values, table.loc[table.window, 'active'].values)
    per_label = ev.per_label(table, ACTIVE_FEATURE, pooled).reset_index().assign(threshold=pooled)
    per_label.to_csv(f'{out}/per_label.csv', index=False)
    print(per_label.round(2).to_string(index=False), '\n')
    distance = pd.concat(distance)
    distance.to_csv(f'{out}/distance.csv', index=False)
    print(distance.pivot(index='dist_m', columns='feature', values='median').round(2).to_string())
