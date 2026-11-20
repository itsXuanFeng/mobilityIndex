"""
Prototype: an online detector of moving cells, to replace the global noise floor (20th percentile over the whole
recording + 6 dB). Causal, frame by frame on the cube:

1. dynamic clutter map per range-Doppler cell: a running Q-quantile of the power in dB. Per frame it moves up by
   eta * Q when the cell is above it and down by eta * (1 - Q) otherwise, with eta set so that a reflector staying
   10 dB above the map is absorbed after tau. A person present in a cell less than (1 - Q) of the time does not
   raise it. Started from the Q-quantile of the first INIT_S seconds.
2. the current map minus the clutter map, in dB (Z = P / map in linear power);
3. 2D SO-CFAR on Z: background = the smallest of the four training strips (range lag / lead, Doppler lag / lead)
   outside a guard window; a cell is detected when Z > k x background.

Lanes, scored side by side:
    q<tau>_so<k>     running quantile + SO-CFAR
    q<tau>_fixed6    running quantile + fixed 6 dB margin, no CFAR: the current method made online
    raw_so<k>        SO-CFAR on the map itself, no clutter map
    mean120_so6      clutter map as a running mean (tau 120 s, detected cells 10 x slower) + SO-CFAR. Fails: the
                     person's echoes raise the mean where they move, and SO picks those low-ratio strips.
    n_cells          the current detector (cached; global 20th percentile + 6 dB)
Each detects moving cells (|v| >= MIN_SPEED, r >= MIN_RANGE) and counts them per frame, like n_cells, scored as in
scripts/evaluate_category1.py (radar threshold leave-one-subject-out, AUC and F1 per recording vs MoCap active).
Needs scripts/evaluate_category1.py to have run.

Two risks of an online map are scored on their own, F1 pooled over the epochs (in the evaluation window):
    in bed       the lying period, from "Lay down on the bed" to "Get up and sit on the edge of the bed" (turns, lying
                 still, the wearable button included): a person staying in the same cells could be absorbed by the map
    cold start   convergence: every lane is run a second time, started cold at "Walk towards the chair" (the arm tasks
                 at the chair follow), and scored on the WARMUP_S after that restart against the same epochs of the
                 converged run, with the same thresholds (those of the converged run)

    python scripts/prototype_so_cfar.py --workers 16
    python scripts/prototype_so_cfar.py --only person_1/scenario_1      # quick check, no scores

Writes so_cfar_prototype.csv to its run folder, output/experiments/so_cfar_<time>/.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility import evaluation as ev  # noqa: E402
from mobility.cli import parallel_map, wide_tables  # noqa: E402
from mobility.config import EPOCH_S, FPS  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.groundtruth import active_speed_thresholds  # noqa: E402
from mobility.radar.cube import load_cube  # noqa: E402
from mobility.radar.features import MOVING  # noqa: E402
from mobility.session import load_session  # noqa: E402

TAUS_S = [30, 120, 600]       # clutter-map time constant
Q = 0.2                       # clutter map: running quantile of each cell (as the current 20th percentile)
INIT_S = 2                    # the map starts as the Q-quantile of the first INIT_S seconds
K_DB = [3, 4.5, 6]            # CFAR scale factor
FIXED_DB = 6                  # margin of the no-CFAR lanes (current ABOVE_FLOOR_DB)
GUARD = (4, 3)                # guard cells on each side: range (4.4 cm), Doppler (4.3 cm/s)
TRAIN = (8, 8)                # training cells per strip: range, Doppler
WIDTH = 3                     # strips are WIDTH cells wide across the other axis
WARMUP_S = 120                # cold start: scored on this long after the restart (clutter map converging)
RESTART_LABEL = 'walk towards the chair'                                       # cold start here
BED_SPAN = ('lay down on the bed', 'get up and sit on the edge of the bed')   # lying period: first to last label
MEAN_TAU_S, MEAN_SLOW = 120, 10
SOURCES = ['raw', 'mean'] + [f'q{t}' for t in TAUS_S]
# lanes: (name, source map, CFAR k in dB or None = fixed margin)
LANES = ([(f'q{t}_so{k:g}', f'q{t}', k) for t in TAUS_S for k in K_DB] + [(f'q{t}_fixed6', f'q{t}', None) for t in TAUS_S]
         + [(f'raw_so{k:g}', 'raw', k) for k in K_DB] + [('mean120_so6', 'mean', 6)])


def strips(Z, axis, guard, n, mode):
    """mean of the n cells before (lag) and after (lead) the guard cells along axis (1 = range, 2 = Doppler)"""
    pad = guard + n
    Zp = np.pad(Z, [(pad, pad) if ax == axis else (0, 0) for ax in range(Z.ndim)], mode=mode)
    c = np.cumsum(Zp, axis=axis)
    c = np.concatenate([np.zeros_like(c.take([0], axis=axis)), c], axis=axis)     # c[j] = sum of Zp[:j]
    i = np.arange(Z.shape[axis]) + pad
    lag = (c.take(i - guard, axis=axis) - c.take(i - guard - n, axis=axis)) / n
    lead = (c.take(i + guard + 1 + n, axis=axis) - c.take(i + guard + 1, axis=axis)) / n
    return lag, lead


def so_background(Z):
    """SO-CFAR background of every cell: the smallest of its four training-strip means. Z: (lanes, range, Doppler);
    Doppler wraps around (FFT), range is mirrored at its ends"""
    zr = uniform_filter1d(Z, WIDTH, axis=2, mode='wrap')        # range strips, WIDTH Doppler bins wide
    zd = uniform_filter1d(Z, WIDTH, axis=1, mode='reflect')     # Doppler strips, WIDTH range bins wide
    r_lag, r_lead = strips(zr, 1, GUARD[0], TRAIN[0], 'reflect')
    d_lag, d_lead = strips(zd, 2, GUARD[1], TRAIN[1], 'wrap')
    return np.minimum(np.minimum(r_lag, r_lead), np.minimum(d_lag, d_lead))


def detections(cube):
    """(lanes, T) number of detected moving cells per frame; NaN for frames without data and the first INIT_S s"""
    T, n0 = len(cube), INIT_S * FPS
    eta = np.array([10 / (Q * t * FPS) for t in TAUS_S], np.float32)[:, None, None]   # dB per frame
    mean_rate = 1 / (MEAN_TAU_S * FPS)
    out = np.full((len(LANES), T), np.nan)
    first, q_map, mean_map, seen = [], None, None, 0
    for a in range(0, T, 2000):
        for j, x in enumerate(np.asarray(cube[a:a + 2000], np.float32)):
            if np.isnan(x).all():
                continue
            P = 10 ** (x / 10)
            if q_map is None:                           # start: collect the first INIT_S s
                first.append(x)
                if len(first) == n0:
                    q_map = np.repeat(np.percentile(first, 100 * Q, axis=0)[None], len(TAUS_S), 0)
                    mean_map = np.mean(10 ** (np.array(first) / 10), axis=0)
                    seen = n0
                continue
            Z = dict(raw=P, mean=P / mean_map)
            Z.update({f'q{t}': 10 ** ((x - q_map[i]) / 10) for i, t in enumerate(TAUS_S)})
            bg = dict(zip(SOURCES, so_background(np.stack([Z[k] for k in SOURCES]))))
            det = {}
            for i, (name, src, k) in enumerate(LANES):
                det[name] = Z[src] > (10 ** (k / 10) * bg[src] if k is not None else 10 ** (FIXED_DB / 10))
                out[i, a + j] = (det[name] & MOVING).sum()
            # update the maps after the test, so a frame is never judged against itself
            q_map += eta * np.where(x[None] > q_map, Q, Q - 1)
            w = max(mean_rate, 1 / (seen + 1))
            mean_map += np.where(det['mean120_so6'], w / MEAN_SLOW, w) * (P - mean_map)
            seen += 1
    return out


def _label_time(labels, text, which='start', after=None):
    """start or end (epoch s) of the first label with this text (at or after `after`), or None"""
    lb = labels[labels.label.str.lower() == text]
    if after is not None:
        lb = lb[lb.start >= after]
    return float(lb[which].iloc[0]) if len(lb) else None


def epochs(rec, gt_thr):
    """the evaluation rows of ev.epoch_table for one recording, with the epoch level of every lane, of every lane
    started cold at RESTART_LABEL (<lane>_cold), and the in_bed and cold_start subsets"""
    person, scenario = rec.split('/')
    s = load_session(person, scenario)
    cube = load_cube(person, scenario)
    counts = {name: x for (name, _, _), x in zip(LANES, detections(cube))}
    restart = _label_time(s.labels, RESTART_LABEL)
    if restart is not None:
        r0 = max(int(round((restart - s.t[0]) * FPS)), 0)
        cold = detections(cube[r0:r0 + (WARMUP_S + 2 * INIT_S) * FPS])
        for (name, _, _), x in zip(LANES, cold):
            counts[f'{name}_cold'] = np.full(len(cube), np.nan)
            counts[f'{name}_cold'][r0:r0 + len(x)] = x
    df = ev.count_epoch_table(s, counts, gt_thr[person], with_label=True)  # with n_cells: the global floor
    t = s.t[0] + df.t_s.values                                                # epoch start, epoch s
    bed0 = _label_time(s.labels, BED_SPAN[0])
    bed1 = _label_time(s.labels, BED_SPAN[1], 'end', after=bed0) if bed0 is not None else None
    df['in_bed'] = (bed0 is not None) & (bed1 is not None) & (t + EPOCH_S / 2 >= (bed0 or 0)) & \
        (t + EPOCH_S / 2 <= (bed1 or 0))
    df['cold_start'] = (restart is not None) & (t >= (restart or 0) + INIT_S + EPOCH_S) & \
        (t + EPOCH_S <= (restart or 0) + WARMUP_S)
    return df


def _epochs(args):
    return epochs(*args)


def pooled_f1(w, feature, thresholds, subset):
    g = w[subset & w[feature].notna()]
    return f1_score(g.active, g[feature].values > g.rec.map(thresholds).values)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', metavar='PERSON/SCENARIO', help='time one recording, no scores')
    add_run_args(ap)
    args = ap.parse_args()
    with run('so_cfar', args, partial=bool(args.only)) as run_dir:
        gt_thr = active_speed_thresholds()
        if args.only:
            t0 = time.time()
            d = epochs(args.only, gt_thr)
            print(f'{args.only}: {len(d)} epochs in {time.time() - t0:.0f} s')
            print(d[[c for c, _, _ in LANES] + ['n_cells']].describe().round(2).to_string())
            sys.exit()
        recs = pd.read_csv(latest('category1', 'thresholds.csv')).rec.tolist()   # evaluated ones
        table = pd.concat(parallel_map(_epochs, [(r, gt_thr) for r in recs], args.workers), ignore_index=True)
        w = table[table.window]
        print(f'{table.rec.nunique()} recordings, {len(w)} epochs in the evaluation window')
        rows = []
        print(f'in bed: {w.in_bed.sum()} epochs; cold start: {w.cold_start.sum()} epochs')
        for name, tau, kd in [('n_cells', None, None)] + LANES:
            thr = ev.loso_thresholds(table, name, gt_thr)
            sc = ev.epoch_scores(table, name, thr)
            cold = f'{name}_cold'
            rows.append(dict(detector=name, tau_s=tau, k_db=kd, auc_median=sc.auc.median(), f1_median=sc.f1.median(),
                             f1_p10=sc.f1.quantile(0.1), f1_in_bed=pooled_f1(w, name, thr, w.in_bed),
                             f1_converged=pooled_f1(w, name, thr, w.cold_start),
                             f1_cold_start=pooled_f1(w, cold, thr, w.cold_start) if cold in w else np.nan))
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(run_dir, 'so_cfar_prototype.csv'), index=False)
        wide_tables()
        print(res.round(3).to_string(index=False))
