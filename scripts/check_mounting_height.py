"""
The walking moments (mobility.moments) and what they say about the radar's installed height, on the recordings of
Category 1, leaving each person out. Radar data only, except MoCap as the reference for the purity and the stand-in
for a typed-in stature.

1. Purity of the walking moments: for a grid of speed and path settings, the share of walking frames where MoCap
   says standing (MoCap posture rule) and of walking epochs MoCap calls active, and the walking time per recording.
2. Walking height: per recording and per person (all their recordings), the median residual of the radar's
   top-of-body height on walking frames from the standing mean of the posture model (calibrated without the person;
   latest category3 run). The frames used are the resident's: inside the script, without the caregiver.
   Distance correction (mobility.moments.DistanceCurve): the residual's median per 0.25 m of horizontal distance from
   the radar axis, each person's own median removed, fitted on the other persons' walking frames and subtracted. Three
   corrections are compared: none (the posture model's standing line only); the distance curve; and, for single
   recordings, the distance curve fitted without the recording's furniture layout too (a new room).
3. Mounting check (mobility.moments.MountingCheck). An error delta in the installed radar height moves every
   measured height by delta, so it moves the walking-height offset by delta. Per left-out person, from the other
   persons: the reference (median offset) and the limit (the ALPHA quantile of |offset - reference|: the share of
   correctly installed radars flagged). Then the person's offset + delta for each simulated error is checked:
   detection rate per delta, and the error of the installed height estimated as offset - reference.
   Two amounts of data: one recording (~10 min, about 20 s of walking) and all the person's recordings (~1 h).
   Two variants: walking height alone; and with the resident's stature known (typed in at installation; here the
   MoCap head height while walking stands in for it): the offset is first predicted from the stature by a line
   fitted on the other persons, and the check acts on the rest. The stand-in is close to what the radar sees, so a
   third variant adds a measurement error of STATURE_NOISE_SD to every stature (a tape measure, typed in), the same
   for calibration and test, one error per person, averaged over STATURE_DRAWS random draws.

    python scripts/check_mounting_height.py --workers 32

Writes to its run folder, output/experiments/mounting_check_<time>/:
    purity.csv          per walking setting: frames, standing share, active share, walking seconds per recording
    recordings.csv      per recording: walking seconds, offset (top, median height), stature stand-in, purity
    persons.csv         per person: walking seconds, offset, stature stand-in (all recordings)
    distance_curve.csv  per left-out person (and ALL): the distance curve (bin centre d, value)
    by_distance.csv     walking frames per 0.25 m of distance: count, person-centred residual median and quartiles
    mounting_check.csv  per left-out person (and ALL), amount of data and correction: reference, limit, min_seconds;
                        stature line (intercept, slope) and its limit
    detection.csv       per amount of data, correction, variant and simulated error: share flagged, estimate error
    mounting_check.png  the walking height against distance; the offsets per person with the check band; the
                        detection rate per error (distance curve)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.cli import error_summary, parallel_map  # noqa: E402
from mobility.config import EPOCH_FRAMES, FPS  # noqa: E402
from mobility.evaluation.agreement import paired_states  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.mounting import plot_mounting_check  # noqa: E402
from mobility.moments import (DISTANCE_BINS, DISTANCE_CURVE_FILE, MOUNTING_CHECK_FILE, WALK_MIN_PATH,  # noqa: E402
                              WALK_SPEED, fit_distance_curve, session_walking, walking_residuals)
from mobility.posture import (STANDING, UNKNOWN, posture_rules, radar_height, radar_median_height,  # noqa: E402
                              radar_offaxis, radar_posture_models)
from mobility.session import load_session  # noqa: E402
from mobility.utils import epoch_mean, fold_of, spread  # noqa: E402

SPEEDS = (0.3, 0.4, 0.5)          # m/s, purity grid
PATHS = (0.5, 1.0, 1.5)           # m
ALPHA = 0.05                      # share of correctly installed radars the check may flag
MIN_SECONDS = 10.0                # s of walking before the check says anything
DELTAS = (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)   # m, simulated installed-height errors
DATA = ('recording', 'all')
STATURE_NOISE_SD = 0.02           # m: a typed-in stature measured with this error (variant 'with stature +-2 cm')
STATURE_DRAWS = 50                # random draws of those errors, averaged (27 persons: one draw is noisy)
STATURE_VARIANTS = {'with stature': 'stature', 'with stature +-2 cm': 'stature_noisy'}
CORRECTIONS = {'none': 'offset_none', 'distance curve': 'offset_curve',
               'distance curve, layout out': 'offset_curve_layout_out'}     # the last: single recordings only

MODELS, MRULES = {}, {}


def per_recording(row):
    """one recording: purity per walking setting, walking residuals (default setting), stature stand-in"""
    try:
        s = load_session(*parse_recording(row.rec))
        mrule = fold_of(MRULES, row.person)
        rs, ms = paired_states(s, row.radar_threshold, row.mocap_threshold, active_feature=row.active_feature,
                               energy_feature=row.energy_feature)
        E = len(rs.t)
        resident = s.masks['in_script'] & s.masks['no_caregiver']      # the resident alone (no operator, caregiver)
        mp = mrule.classify(s.mocap)
        known = s.masks['window'] & (mp != UNKNOWN)
        purity = []
        for v in SPEEDS:
            for p in PATHS:
                w = session_walking(s, speed=v, min_path=p) & resident
                we = (epoch_mean(w, EPOCH_FRAMES)[:E] >= 0.5) & ms.valid
                purity.append(dict(speed=v, path=p, frames=int((w & known).sum()),
                                   standing=int((w & known & (mp == STANDING)).sum()), epochs=int(we.sum()),
                                   active=int(ms.active[we].sum()), walking_s=w.sum() / FPS))
        w = session_walking(s) & resident
        res = walking_residuals(radar_height(s), radar_median_height(s), radar_offaxis(s), w,
                                fold_of(MODELS, row.person), with_offaxis=True)
        head = np.asarray(s.mocap.get('head_y', np.full(len(s.t), np.nan)))
        hw = head[w & (mp == STANDING) & ~np.isnan(head)]
        return dict(rec=row.rec, person=row.person, scenario=s.scenario, purity=purity, residuals=res,
                    head=float(np.median(hw)) if len(hw) else np.nan), None
    except Exception:
        return None, f'{row.rec}: {error_summary()}'


def frame_table(recs):
    """all walking frames: recording, person, scenario, distance from the axis, top residual"""
    return pd.concat([pd.DataFrame(dict(rec=d['rec'], person=d['person'], scenario=d['scenario'],
                                        d=d['residuals'][:, 2], r=d['residuals'][:, 0])) for d in recs],
                     ignore_index=True)


def curves(frames):
    """the distance curve per left-out person (and 'ALL'), and per left-out (person, scenario)"""
    fit = lambda f: fit_distance_curve(f.d.values, f.r.values, f.person.values)   # noqa: E731
    persons = sorted(frames.person.unique())
    per = {p: fit(frames[frames.person != p]) for p in persons}
    per['ALL'] = fit(frames)
    lay = {(p, sc): fit(frames[(frames.person != p) & (frames.scenario != sc)])
           for p, sc in frames[['person', 'scenario']].drop_duplicates().itertuples(index=False)}
    return per, lay


def levels(recs, frames, per_curve, layout_curve):
    """per recording and per person: walking seconds, offsets under each correction, stature stand-in"""
    f = frames.copy()
    f['none'] = f.r
    f['curve'] = f['curve_lo'] = np.nan
    for p, idx in f.groupby('person').groups.items():
        f.loc[idx, 'curve'] = f.r[idx] - per_curve[p](f.d[idx].values)
    for (p, sc), idx in f.groupby(['person', 'scenario']).groups.items():
        f.loc[idx, 'curve_lo'] = f.r[idx] - layout_curve[p, sc](f.d[idx].values)
    head = {d['rec']: d['head'] for d in recs}
    rec = f.groupby('rec').agg(person=('person', 'first'), frames=('r', 'size'), offset_none=('none', 'median'),
                               offset_curve=('curve', 'median'), offset_curve_layout_out=('curve_lo', 'median'),
                               d_median=('d', 'median')).reset_index()
    rec['walking_s'] = rec.frames / FPS
    rec['stature'] = rec.rec.map(head)
    missing = [d['rec'] for d in recs if d['rec'] not in set(rec.rec)]
    if missing:
        rec = pd.concat([rec, pd.DataFrame(dict(rec=missing, person=[r.split('/')[0] for r in missing], frames=0,
                                                walking_s=0.0, stature=[head[r] for r in missing]))],
                        ignore_index=True)
    per = f.groupby('person').agg(frames=('r', 'size'), offset_none=('none', 'median'),
                                  offset_curve=('curve', 'median')).reset_index()
    per['walking_s'] = per.frames / FPS
    per['recordings'] = per.person.map(rec.groupby('person').size())
    per['stature'] = per.person.map(rec.groupby('person').stature.median())
    return rec.drop(columns='frames'), per.drop(columns='frames')


def calibrate(units, exclude, col, variants=STATURE_VARIANTS):
    """reference, limit; per stature variant, the stature line and its limit; from the units (recordings or
    persons) of every person but `exclude` with enough walking, on the offsets `col`"""
    u = units[(units.person != exclude) & (units.walking_s >= MIN_SECONDS) & units[col].notna()]
    ref = float(np.median(u[col]))
    lim = float(np.quantile(np.abs(u[col] - ref), 1 - ALPHA))
    out = dict(reference=ref, limit=lim, min_seconds=MIN_SECONDS, n_units=len(u))
    for name, scol in variants.items():
        s = u[u[scol].notna()]
        slope, icpt = np.polyfit(s[scol], s[col], 1)
        key = name.replace('with ', '').replace(' +-2 cm', '_noisy')
        out.update({f'{key}_intercept': float(icpt), f'{key}_slope': float(slope),
                    f'{key}_limit': float(np.quantile(np.abs(s[col] - (icpt + slope * s[scol])), 1 - ALPHA))})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=1)
    add_run_args(ap)
    args = ap.parse_args()
    with run('mounting_check', args) as out:
        th = pd.read_csv(latest('category1', 'thresholds.csv'))
        MODELS.update(radar_posture_models())
        MRULES.update(posture_rules())
        res = parallel_map(per_recording, [r for _, r in th.iterrows()], args.workers)
        failed = [e for _, e in res if e]
        if failed:
            print(f'{len(failed)} recordings FAILED and are left out:\n  ' + '\n  '.join(failed))
        recs = [d for d, e in res if not e]
        print(f'{len(recs)} recordings, {len({d["person"] for d in recs})} persons', flush=True)

        # 1. purity
        pu = pd.DataFrame([dict(rec=d['rec'], **p) for d in recs for p in d['purity']])
        purity = pu.groupby(['speed', 'path']).agg(frames=('frames', 'sum'), standing=('standing', 'sum'),
                                                   epochs=('epochs', 'sum'), active=('active', 'sum'),
                                                   walking_s_median=('walking_s', 'median'),
                                                   walking_s_p10=('walking_s', lambda x: np.percentile(x, 10)))
        purity['standing_share'] = purity.standing / purity.frames
        purity['active_share'] = purity.active / purity.epochs
        purity = purity.reset_index()
        purity.to_csv(f'{out}/purity.csv', index=False)
        print(f'\n1. walking moments against MoCap (chosen: speed {WALK_SPEED} m/s, path {WALK_MIN_PATH} m over the '
              f'trailing 3 s):\n{purity.round(4).to_string(index=False)}')

        # 2. walking height, and its dependence on the distance from the axis
        frames = frame_table(recs)
        per_curve, layout_curve = curves(frames)
        pd.DataFrame([dict(person=p, d=d, value=v) for p, c in per_curve.items() for d, v in zip(c.centres, c.values)]
                     ).to_csv(f'{out}/{DISTANCE_CURVE_FILE}', index=False)
        fc = frames.assign(rc=frames.r - frames.groupby('person').r.transform('median'))
        g = fc.groupby(pd.cut(fc.d, DISTANCE_BINS), observed=True).rc
        by_d = pd.DataFrame(dict(frames=g.size(), share=g.size() / len(fc), median=g.median(), q25=g.quantile(0.25),
                                 q75=g.quantile(0.75))).reset_index()
        by_d.insert(1, 'd_hi', [iv.right for iv in by_d.d])
        by_d.insert(1, 'd_lo', [iv.left for iv in by_d.d])
        by_d['d'] = by_d.d.astype(str)
        by_d.to_csv(f'{out}/by_distance.csv', index=False)
        rec, per = levels(recs, frames, per_curve, layout_curve)
        rec.to_csv(f'{out}/recordings.csv', index=False)
        per.to_csv(f'{out}/persons.csv', index=False)
        print(f'\n2. walking height: walking per recording {spread(rec.walking_s, ".0f")} s (median [10th-90th]), '
              f'per person {spread(per.walking_s, ".0f")} s; recordings with < {MIN_SECONDS:g} s: '
              f'{(rec.walking_s < MIN_SECONDS).sum()}')
        print(f'   top-of-body residual on walking frames by distance from the radar axis, each person\'s median '
              f'removed (m):\n{by_d.round(3).to_string(index=False)}')
        print(f'   distance curve, all persons: ' + ', '.join(f'{d:.3g} m {v:+.3f}' for d, v in
                                                          zip(per_curve['ALL'].centres, per_curve['ALL'].values)))
        good = rec[rec.walking_s >= MIN_SECONDS]
        for name, col in CORRECTIONS.items():
            pp = f', per person {spread(per[col])} m (sd {per[col].std():.3f})' if col in per else ''
            print(f'   offset, {name:26s}: per recording {spread(good[col])} m{pp}; within a person, sd over '
                  f'recordings {good.groupby("person")[col].std().median():.4f} m (median)')
        st = per.dropna(subset=['stature'])
        print(f'   stature stand-in (MoCap head height while walking) {spread(st.stature, ".2f")} m; correlation with '
              f'the offset {np.corrcoef(st.stature, st.offset_curve)[0, 1]:.2f}')

        # 3. the check, leaving each person out
        rng = np.random.default_rng(0)
        noise = rng.normal(0, STATURE_NOISE_SD, (STATURE_DRAWS, len(per)))   # one measurement per person and draw
        units = {'recording': rec, 'all': per}
        cal_rows, det_rows = [], []
        persons = sorted(per.person)
        for data, U in units.items():
            for correction, col in CORRECTIONS.items():
                if col not in U:
                    continue
                for draw in range(STATURE_DRAWS):
                    U['stature_noisy'] = U.stature + U.person.map(dict(zip(per.person, noise[draw])))
                    # walking height alone and exact stature: deterministic, once; the noisy stature: every draw
                    variants = STATURE_VARIANTS if draw == 0 else {k: v for k, v in STATURE_VARIANTS.items()
                                                                   if v == 'stature_noisy'}
                    for p in persons + [None]:
                        c = calibrate(U, p, col, variants)
                        cal_rows.append(dict(person=p or 'ALL', data=data, correction=correction, draw=draw, **c))
                        if p is None:
                            continue
                        own = U[(U.person == p) & (U.walking_s >= MIN_SECONDS) & U[col].notna()]
                        for o in own.itertuples():
                            for delta in DELTAS:
                                x = getattr(o, col) + delta
                                base = dict(data=data, correction=correction, person=p, unit=getattr(o, 'rec', p),
                                            delta=delta, draw=draw)
                                if draw == 0:
                                    det_rows.append(dict(base, variant='walking height',
                                                         flagged=abs(x - c['reference']) > c['limit'],
                                                         estimate_error=(x - c['reference']) - delta))
                                for name, scol in variants.items():
                                    key = name.replace('with ', '').replace(' +-2 cm', '_noisy')
                                    st = getattr(o, scol)
                                    if np.isnan(st):
                                        continue
                                    pred = c[f'{key}_intercept'] + c[f'{key}_slope'] * st
                                    det_rows.append(dict(base, variant=name,
                                                         flagged=abs(x - pred) > c[f'{key}_limit'],
                                                         estimate_error=(x - pred) - delta))
        cal = pd.DataFrame(cal_rows)
        # one row per person, data and correction: the deterministic values of draw 0, the noisy-stature line and
        # limit averaged over the draws
        noisy = [c for c in cal.columns if c.startswith('stature_noisy')]
        keys = ['person', 'data', 'correction']
        cal = cal[cal.draw == 0].drop(columns=noisy + ['draw']).merge(
            cal.groupby(keys)[noisy].mean().reset_index(), on=keys)
        cal.to_csv(f'{out}/{MOUNTING_CHECK_FILE}', index=False)
        det = pd.DataFrame(det_rows)
        summary = det.groupby(['data', 'correction', 'variant', 'delta']).agg(
            n=('flagged', 'size'), flagged=('flagged', 'mean'),
            estimate_mae=('estimate_error', lambda e: e.abs().mean()),
            estimate_p90=('estimate_error', lambda e: e.abs().quantile(0.9)))
        summary = summary.reset_index()
        summary.to_csv(f'{out}/detection.csv', index=False)
        held = cal[cal.person != 'ALL']
        print(f'\n3. mounting check, leaving each person out (flag at {100 * (1 - ALPHA):.0f}% of the other persons):')
        for (data, corr), g in held.groupby(['data', 'correction'], sort=False):
            print(f'   {data:9s} {corr:26s} limit +-{spread(g.limit)} m; with stature +-{spread(g.stature_limit)} m; '
                  f'stature +-2 cm: +-{spread(g.stature_noisy_limit)} m')
        wide = summary.pivot_table(index=['data', 'variant', 'correction'], columns='delta', values='flagged')
        wide.columns = [f'{c:+.2f}' for c in wide.columns]
        print(f'   share flagged per simulated error of the installed height (m); column +0.00 = false alarms:\n'
              f'{wide.round(2).to_string()}')
        est = summary[summary.delta == 0].set_index(['data', 'variant', 'correction'])[['estimate_mae',
                                                                                         'estimate_p90']]
        print(f'   error of the installed height estimated from the walking height (m):\n{est.round(3).to_string()}')

        main = held[(held.data == 'all') & (held.correction == 'distance curve')]
        plot_mounting_check(per, main, summary[summary.correction == 'distance curve'], by_d, per_curve['ALL'],
                            f'{out}/mounting_check.png')
