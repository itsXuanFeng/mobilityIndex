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
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.config import ABOVE_FLOOR_DB, EPOCH_S, FLOOR_PERCENTILE, FPS, OUTPUT_ROOT  # noqa: E402
from mobility.groundtruth import active_speed_thresholds  # noqa: E402
from mobility.plots import style  # noqa: E402
from mobility.radar.cube import load_cube  # noqa: E402
from mobility.radar.features import MOVING  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.states import mocap_states  # noqa: E402
from mobility.utils import epoch_mean, log_level  # noqa: E402

PERCENTILES = [5, 10, 20, 35, 50]
MARGINS_DB = [3, 4.5, 6, 8, 10, 12]
TABLE = os.path.join(OUTPUT_ROOT, 'category1', 'floor_sensitivity.csv')
FIGURE = os.path.join(OUTPUT_ROOT, 'plots', 'sensitivity', 'floor_sensitivity.png')
# sequential blue ramp of the reference palette (steps 100-700)
SCALE = (0.80, 1.00)     # one colour scale for every panel, so small differences look small
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


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
    n = int(round(EPOCH_S * FPS))
    ms = mocap_states(s, active_speed=gt_thr[person], window='has_mocap')
    E = len(ms.t)
    df = pd.DataFrame(dict(rec=s.name, person=person, speed=ms.level, active=ms.active,
                           window=epoch_mean(s.masks['window'], n)[:E] >= 0.5))
    for c, x in counts.items():
        df[c] = epoch_mean(log_level(x), n)[:E]          # as states.feature_level for a log feature
    df['n_cells'] = epoch_mean(log_level(s.radar['n_cells']), n)[:E]   # cached feature, to check the default
    return df[ms.valid]


def _epochs(args):
    return epochs(*args)


def sweep(workers):
    recs = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv')).rec.tolist()   # evaluated ones
    gt_thr = active_speed_thresholds()
    with Pool(workers) as pool:
        table = pd.concat(pool.map(_epochs, [(r, gt_thr) for r in recs]), ignore_index=True)
    print(f'{table.rec.nunique()} recordings, {table.window.sum()} epochs in the evaluation window')
    rows = []
    for p in PERCENTILES:
        for m in MARGINS_DB:
            sc = ev.epoch_scores(table, column(p, m), ev.loso_thresholds(table, column(p, m), gt_thr))
            rows.append(dict(percentile=p, margin_db=m, recordings=sc.rec.nunique(), auc_median=sc.auc.median(), f1_median=sc.f1.median(),
                             f1_p10=sc.f1.quantile(0.1), f1_min=sc.f1.min()))
    default = ev.epoch_scores(table, 'n_cells', ev.loso_thresholds(table, 'n_cells', gt_thr))
    print(f'check, cached n_cells: AUC {default.auc.median():.3f}, F1 {default.f1.median():.3f}')
    out = pd.DataFrame(rows)
    out.to_csv(TABLE, index=False)
    return out


def heatmap(ax, grid, title, fmt='{:.3f}'):
    cmap = LinearSegmentedColormap.from_list('blues', BLUES)
    lo, hi = SCALE
    ax.imshow(grid.values, cmap=cmap, vmin=lo, vmax=hi, aspect='auto', origin='lower')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, p in enumerate(grid.index):
        for j, m in enumerate(grid.columns):
            v = grid.values[i, j]
            shade = cmap(np.clip((v - lo) / (hi - lo), 0, 1))
            dark = 0.2126 * shade[0] + 0.7152 * shade[1] + 0.0722 * shade[2] < 0.45
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color=style.SURFACE if dark else style.INK)
    i, j = list(grid.index).index(FLOOR_PERCENTILE), list(grid.columns).index(ABOVE_FLOOR_DB)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=style.INK, lw=1.8))
    ax.set_xticks(range(len(grid.columns)), [f'{m:g}' for m in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f'{p}th' for p in grid.index])
    ax.set_xlabel('margin above the floor (dB)')
    ax.set_title(title)


def plot(res):
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    panels = [('auc_median', 'AUC, median over recordings'), ('f1_median', 'F1, median over recordings'),
              ('f1_p10', 'F1, 10th percentile (worst recordings)')]
    for ax, (k, title) in zip(axes, panels):
        heatmap(ax, res.pivot(index='percentile', columns='margin_db', values=k), title)
    axes[0].set_ylabel('noise-floor percentile')
    fig.suptitle(f'Active / inactive (n_cells) vs its two parameters, {res.recordings.iloc[0]} recordings, '
                 f'leave-person-out thresholds.\nOutlined: current setting '
                 f'({FLOOR_PERCENTILE}th percentile, {ABOVE_FLOOR_DB:g} dB). Colour: one scale for all panels, '
                 f'{SCALE[0]:.2f} (light) to {SCALE[1]:.2f} (dark).',
                 x=0.01, ha='left', fontsize=9, color=style.INK_2)
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    print(f'wrote {FIGURE}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--plot-only', action='store_true', help='redraw from the saved table')
    args = ap.parse_args()
    res = pd.read_csv(TABLE) if args.plot_only else sweep(args.workers)
    pd.set_option('display.width', 200)
    print(res.round(3).to_string(index=False))
    plot(res)
