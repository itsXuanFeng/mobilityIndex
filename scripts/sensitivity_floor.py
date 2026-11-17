"""
Sensitivity of the active / inactive call (n_cells) to its two hand-set parameters: the noise-floor percentile
(FLOOR_PERCENTILE) and the margin above the floor (ABOVE_FLOOR_DB). For every combination, n_cells is recomputed from
the cubes and scored exactly as in scripts/evaluate_category1.py: radar threshold calibrated leave-one-subject-out,
AUC and F1 per recording against MoCap active. Needs scripts/evaluate_category1.py to have run (evaluated recordings,
session caches).

    python scripts/sensitivity_floor.py --workers 16      # reads all cubes once, ~10 min
    python scripts/sensitivity_floor.py --plot-only       # redraw from the saved table

Writes OUTPUT_ROOT/category1/floor_sensitivity.csv and OUTPUT_ROOT/plots/sensitivity/floor_sensitivity.png.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.cli import parallel_map, wide_tables  # noqa: E402
from mobility.config import CATEGORY1_THRESHOLDS, PLOTS_DIR, category_dir  # noqa: E402
from mobility.groundtruth import active_speed_thresholds  # noqa: E402
from mobility.plots.sensitivity import floor_sensitivity_figure  # noqa: E402
from mobility.radar.cube import load_cube  # noqa: E402
from mobility.radar.features import MOVING  # noqa: E402
from mobility.session import load_session  # noqa: E402

PERCENTILES = [5, 10, 20, 35, 50]
MARGINS_DB = [3, 4.5, 6, 8, 10, 12]
TABLE = os.path.join(category_dir(1), 'floor_sensitivity.csv')
FIGURE = os.path.join(PLOTS_DIR, 'sensitivity', 'floor_sensitivity.png')


def column(p, m):
    return f'p{p}_m{m:g}'


def epochs(rec, gt_thr):
    """the evaluation rows of ev.epoch_table for one recording, with the epoch level of n_cells for every
    (percentile, margin)"""
    person, scenario = rec.split('/')
    s = load_session(person, scenario)
    cube = load_cube(person, scenario)
    floors = np.nanpercentile(np.asarray(cube[::10], np.float32), PERCENTILES, axis=0)   # as noise_floor()
    T = len(cube)
    counts = {column(p, m): np.full(T, np.nan) for p in PERCENTILES for m in MARGINS_DB}
    for a in range(0, T, 2000):
        db = np.asarray(cube[a:a + 2000], np.float32)
        no_data = np.isnan(db).all((1, 2))
        for p, floor in zip(PERCENTILES, floors):
            above = np.where(MOVING, db - floor, -np.inf)
            for m in MARGINS_DB:
                x = (above > m).sum((1, 2)).astype(float)
                x[no_data] = np.nan
                counts[column(p, m)][a:a + len(db)] = x
    return ev.count_epoch_table(s, counts, gt_thr[person])   # with the cached n_cells, to check the default


def _epochs(args):
    return epochs(*args)


def sweep(workers):
    recs = pd.read_csv(CATEGORY1_THRESHOLDS).rec.tolist()   # evaluated ones
    gt_thr = active_speed_thresholds()
    table = pd.concat(parallel_map(_epochs, [(r, gt_thr) for r in recs], workers), ignore_index=True)
    print(f'{table.rec.nunique()} recordings, {table.window.sum()} epochs in the evaluation window')
    rows = []
    for p in PERCENTILES:
        for m in MARGINS_DB:
            sc = ev.epoch_scores(table, column(p, m), ev.loso_thresholds(table, column(p, m), gt_thr))
            rows.append(dict(percentile=p, margin_db=m, recordings=sc.rec.nunique(), auc_median=sc.auc.median(),
                             f1_median=sc.f1.median(), f1_p10=sc.f1.quantile(0.1), f1_min=sc.f1.min()))
    default = ev.epoch_scores(table, 'n_cells', ev.loso_thresholds(table, 'n_cells', gt_thr))
    print(f'check, cached n_cells: AUC {default.auc.median():.3f}, F1 {default.f1.median():.3f}')
    out = pd.DataFrame(rows)
    out.to_csv(TABLE, index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--plot-only', action='store_true', help='redraw from the saved table')
    args = ap.parse_args()
    res = pd.read_csv(TABLE) if args.plot_only else sweep(args.workers)
    wide_tables()
    print(res.round(3).to_string(index=False))
    floor_sensitivity_figure(res, FIGURE)
    print(f'wrote {FIGURE}')
