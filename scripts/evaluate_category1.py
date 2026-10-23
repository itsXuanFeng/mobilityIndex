"""
Category 1 (total movement volume): radar vs MoCap on the recordings that have a cube, MoCap and labels.
Needs the MoCap active thresholds: run scripts/calibrate_mocap_threshold.py first.

    python scripts/evaluate_category1.py --workers 16                  # every recording with a cube
    python scripts/evaluate_category1.py --only person_1/scenario_1 person_5/scenario_3

All thresholds are calibrated leave-one-subject-out. Writes to OUTPUT_ROOT/category1/:
    alignment.csv      time offset and its correlation per recording
    thresholds.csv     per recording: the radar and MoCap active thresholds used for it
    epoch_scores.csv   per feature and recording: AUC, F1
    metrics.csv        per recording: the Category 1 metrics, radar vs MoCap
    agreement.csv      per metric across recordings: mean absolute error and bias (times), Spearman (energy)
    per_label.csv      per scripted activity: MoCap vs radar active share
    distance.csv       distance dependence of each feature
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
from mobility.config import LABEL_ROOT, OUTPUT_ROOT  # noqa: E402
from mobility.groundtruth import active_speed_thresholds, subject  # noqa: E402
from mobility.io.recordings import find_recordings, parse_recording  # noqa: E402
from mobility.radar.cube import cube_path  # noqa: E402
from mobility.session import load_session  # noqa: E402

FEATURES = ['n_cells', 'p_r4', 'p_sub', 'v_rms']
ACTIVE_FEATURE = 'n_cells'     # decides active / inactive
ENERGY_FEATURE = 'p_r4'        # total activity energy
POOR_ALIGNMENT = 0.1           # flag recordings whose alignment peak stands out less than this


def prepare(job):
    """build the session caches (features, MoCap, alignment); runs in a worker"""
    (person, scenario), overwrite = job
    try:
        load_session(person, scenario, overwrite)
        return None
    except Exception:
        return f'{person}/{scenario}: {traceback.format_exc().strip().splitlines()[-1]}'


def spread(x):
    """median [10th-90th percentile]"""
    q = np.nanpercentile(x, [50, 10, 90])
    return f'{q[0]:.3f} [{q[1]:.3f}-{q[2]:.3f}]'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--workers', type=int, default=1, help='recordings prepared in parallel')
    ap.add_argument('--overwrite', action='store_true', help='recompute cached features, MoCap signals, alignment')
    args = ap.parse_args()
    recs = [parse_recording(r) for r in args.only] if args.only else find_recordings()
    recs = [r for r in recs if os.path.exists(cube_path(*r))]
    usable = [r for r in recs if os.path.exists(os.path.join(LABEL_ROOT, *r, 'timestamps.csv'))
              and subject(*r)['track'] is not None]
    skipped = [r for r in recs if r not in usable]
    if skipped:
        print('skipped, no MoCap or no labels:', ' '.join('/'.join(r) for r in skipped))
    out = os.path.join(OUTPUT_ROOT, 'category1')
    os.makedirs(out, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    jobs = [(r, args.overwrite) for r in usable]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            errors = pool.map(prepare, jobs)
    else:
        errors = [prepare(j) for j in jobs]
    failed = [e for e in errors if e]
    if failed:
        print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
    usable = [r for r, e in zip(usable, errors) if not e]

    gt_thr = active_speed_thresholds()     # per person: MoCap active threshold calibrated without that person
    sessions = [load_session(p, s) for p, s in usable]
    print(f'{len(sessions)} recordings, {len({s.person for s in sessions})} persons')
    al = pd.DataFrame([dict(rec=s.name, shot=s.alignment['shot'], track=os.path.basename(s.alignment['track']),
                            offset_s=s.alignment['offset_s'], corr=s.alignment.get('corr', np.nan),
                            margin=s.alignment.get('margin', np.nan)) for s in sessions])
    al.to_csv(f'{out}/alignment.csv', index=False)
    print(f'alignment: offset {spread(al.offset_s)} s, peak correlation {spread(al["corr"])}, '
          f'margin {spread(al.margin)}')
    poor = al[~(al.margin >= POOR_ALIGNMENT)]
    if len(poor):
        # misaligned recordings would compare unrelated seconds: left out (e.g. person_20, whose MoCap clock is off)
        print(f'{len(poor)} recordings left out, alignment failed (margin < {POOR_ALIGNMENT}):\n'
              f'{poor.round(2).to_string(index=False)}')
        sessions = [s for s in sessions if s.name not in set(poor.rec)]
        print(f'{len(sessions)} recordings, {len({s.person for s in sessions})} persons evaluated')

    table = ev.epoch_table(sessions, FEATURES, gt_thr)
    n_ep = table.groupby('rec').agg(with_mocap=('ke', 'size'), in_window=('window', 'sum'))
    print(f'\nepochs: {len(table)} with MoCap, {table.window.sum()} in the evaluation window '
          f'({100 * table.window.mean():.0f}%); per recording in window {spread(n_ep.in_window)}')

    thresholds, scores, distance = {}, [], []
    for f in FEATURES:
        thresholds[f] = ev.loso_thresholds(table, f, gt_thr)
        scores.append(ev.epoch_scores(table, f, thresholds[f]))
        distance.append(ev.distance_residuals(table, f).reset_index().rename(columns={'index': 'dist_m'})
                        .assign(feature=f))
    scores = pd.concat(scores)
    scores.to_csv(f'{out}/epoch_scores.csv', index=False)
    print('\nper-epoch scores over recordings, median [10th-90th percentile]:')
    for f, g in scores.groupby('feature', sort=False):
        print(f'  {f:8s} AUC {spread(g.auc)}  F1 {spread(g.f1)}')
    worst = scores[scores.feature == ACTIVE_FEATURE].nsmallest(8, 'f1')
    print(f'lowest F1 ({ACTIVE_FEATURE}):\n{worst.round(3).to_string(index=False)}')

    pd.DataFrame([dict(rec=s.name, person=s.person, active_feature=ACTIVE_FEATURE, energy_feature=ENERGY_FEATURE,
                       radar_threshold=thresholds[ACTIVE_FEATURE][s.name], mocap_threshold=gt_thr[s.person])
                  for s in sessions]).to_csv(f'{out}/thresholds.csv', index=False)
    mt = ev.metric_table(sessions, thresholds[ACTIVE_FEATURE], gt_thr, category=1, active_feature=ACTIVE_FEATURE,
                         energy_feature=ENERGY_FEATURE)
    mt.to_csv(f'{out}/metrics.csv', index=False)
    agreement = ev.metric_agreement(mt)
    agreement.to_csv(f'{out}/agreement.csv', index=False)
    print(f'\nper-recording metrics:\n{agreement.round(3).to_string(index=False)}')
    times = mt[mt.metric == 'active_time']
    print(f'active time per recording: MoCap {spread(times.mocap)} min, radar {spread(times.radar)} min')

    pooled = ev.best_threshold(table.loc[table.window, ACTIVE_FEATURE].values, table.loc[table.window, 'active'].values)
    per_label = ev.per_label(table, ACTIVE_FEATURE, pooled).reset_index().assign(threshold=pooled)
    per_label.to_csv(f'{out}/per_label.csv', index=False)
    print(f'\nper scripted activity (pooled threshold {pooled:.2f}):\n{per_label.round(2).to_string(index=False)}')
    distance = pd.concat(distance)
    distance.to_csv(f'{out}/distance.csv', index=False)
    print(f'\ndistance residuals:\n'
          f'{distance.pivot(index="dist_m", columns="feature", values="median").round(2).to_string()}')
