"""
Fit and check the angle (mobility/radar/angle.py) against MoCap, on recordings with angle-pass points
(scripts/angle_pass.py). Per frame with clear movement: the radar's nearest clearly moving reflector (as for r_near and
the radar position fit) against the MoCap joint nearest to the radar (head, shoulders, wrists, knees), when their
ranges agree within --max-range-diff.

Fits the rotation of the array axes about the vertical and whether they are mirrored (put them in mobility/config.py
RADAR_YAW_DEG, RADAR_MIRROR), for both direction estimators, then reports, with the fitted orientation:
    direction error (direction cosines), horizontal position error, and height error: from the angle vs from the
    range alone (radar height - range, what Categories 1-3 use), overall and by distance from the radar axis.

    python scripts/fit_radar_orientation.py --only person_1/scenario_1 person_10/scenario_2

Writes OUTPUT_ROOT/plots/angle/<person>_<scenario>_angle.png: x, z and height over time, radar vs MoCap, with the
phase estimator and the orientation fitted on all the recordings given.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.align import FIT_JOINTS, fit_radar_orientation  # noqa: E402
from mobility.config import OUTPUT_ROOT, RADAR_XYZ  # noqa: E402
from mobility.groundtruth import joint_kinematics  # noqa: E402
from mobility.io.mocap import load_track  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.radar.angle import nearest_reflector, to_room  # noqa: E402
from mobility.radar.points import load_points, points_path  # noqa: E402
from mobility.session import load_session, noise_floor  # noqa: E402

METHODS = ('phase', 'beam')


def frames(rec, max_range_diff, min_joint_speed):
    """per usable frame: radar nearest reflector (both estimators) and the matching MoCap joint"""
    ses = load_session(*rec)
    pts = load_points(*rec)
    floor = noise_floor(*rec)
    t, pos = load_track(ses.alignment['track'], FIT_JOINTS)
    grid, pj, v = joint_kinematics(t, pos)
    out = None
    for method in METHODS:
        nr = nearest_reflector(pts, floor, method)
        k = np.clip(np.searchsorted(grid, ses.t[nr['frame']]), 0, len(grid) - 1)
        rel = pj[k] - np.asarray(RADAR_XYZ)                         # (n, joints, 3)
        dist = np.linalg.norm(rel, axis=-1)
        j = dist.argmin(1)
        joint = rel[np.arange(len(j)), j]
        d = dist[np.arange(len(j)), j]
        speed = np.linalg.norm(v[k], axis=-1).max(1)
        use = (ses.masks['has_mocap'][nr['frame']] & (speed > min_joint_speed)
               & (np.abs(nr['r'] - d) < max_range_diff))
        df = pd.DataFrame({'frame': nr['frame'], f'u_{method}': nr['u'], f'v_{method}': nr['v'], 'r': nr['r'],
                           'jx': joint[:, 0], 'jy': joint[:, 1], 'jz': joint[:, 2], 'jd': d, 'use': use})
        out = df if out is None else out.merge(df[['frame', f'u_{method}', f'v_{method}']], on='frame')
    out = out[out.use].drop(columns='use').assign(rec='/'.join(rec), t=lambda x: ses.t[x.frame] - ses.t[0])
    return out


def errors(df, method, yaw, mirror):
    """per frame: direction, horizontal position and height errors with the given orientation"""
    x, y, z = to_room(df.r.values, df[f'u_{method}'].values, df[f'v_{method}'].values, yaw, mirror)
    jx, jy, jz = df.jx + RADAR_XYZ[0], df.jy + RADAR_XYZ[1], df.jz + RADAR_XYZ[2]
    return pd.DataFrame(dict(horizontal=np.hypot(x - jx, z - jz), height_angle=y - jy,
                             height_range=(RADAR_XYZ[1] - df.r.values) - jy, offaxis=np.hypot(df.jx, df.jz),
                             x=x, y=y, z=z, jx=jx, jy=jy, jz=jz))


def figure(df, err, path, title):
    import matplotlib.pyplot as plt
    from mobility.plots import style
    style.apply()
    fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    t = df.t.values / 60
    for a, (k, label) in zip(ax, (('x', 'x (m)'), ('z', 'z (m)'), ('y', 'height (m)'))):
        a.plot(t, err[f'j{k}'], '.', color=style.MOCAP, markersize=2, label='MoCap: joint nearest the radar')
        a.plot(t, err[k], '.', color=style.RADAR, markersize=2, label='radar: nearest moving reflector, angle')
        a.set_ylabel(label)
    ax[2].plot(t, RADAR_XYZ[1] - df.r.values, '.', color=style.INK_2, markersize=1.5,
               label='radar: radar height - range (no angle)')
    ax[2].legend(loc='lower left', ncol=3)
    ax[2].set_xlabel('minutes since radar start')
    fig.suptitle(title, x=0.06, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO', required=True)
    ap.add_argument('--max-range-diff', type=float, default=0.15, help='m: radar and MoCap ranges must agree')
    ap.add_argument('--min-joint-speed', type=float, default=0.3, help='m/s, fastest joint, to use a frame')
    args = ap.parse_args()
    recs = [parse_recording(r) for r in args.only if os.path.exists(points_path(*parse_recording(r)))]
    df = pd.concat([frames(r, args.max_range_diff, args.min_joint_speed) for r in recs], ignore_index=True)
    print(f'{len(df)} frames from {len(recs)} recordings')
    target = np.c_[df.jx, df.jz] / df.jd.values[:, None]
    fits = {}
    for method in METHODS:
        fits[method] = fit_radar_orientation(np.c_[df[f'u_{method}'], df[f'v_{method}']], target)
        print(f'{method:6s} RADAR_YAW_DEG = {fits[method]["yaw_deg"]}, RADAR_MIRROR = {fits[method]["mirror"]}: '
              f'median direction error {fits[method]["median_err"]:.3f}')
    for method in METHODS:
        e = errors(df, method, fits[method]['yaw_deg'], fits[method]['mirror'])
        print(f'\n{method}: median horizontal error {100 * e.horizontal.median():.0f} cm; median |height error| '
              f'from the angle {100 * e.height_angle.abs().median():.0f} cm (bias {100 * e.height_angle.median():+.0f}),'
              f' from the range alone {100 * e.height_range.abs().median():.0f} cm '
              f'(bias {100 * e.height_range.median():+.0f})')
        by = e.groupby(pd.cut(e.offaxis, [0, 0.5, 1, 1.5, 2, 3, 5])).agg(
            frames=('horizontal', 'size'), horizontal_cm=('horizontal', lambda x: 100 * x.median()),
            height_angle_cm=('height_angle', lambda x: 100 * x.abs().median()),
            height_range_cm=('height_range', lambda x: 100 * x.abs().median()))
        print(f'by distance of the joint from the radar axis (m), medians:\n{by.round(0).to_string()}')
        per = e.groupby(df.rec.values).agg(frames=('horizontal', 'size'),
                                           horizontal_cm=('horizontal', lambda x: 100 * x.median()))
        print(f'per recording:\n{per.round(0).to_string()}')
    best = 'phase'                       # the default of mobility.radar.angle.localise
    out = os.path.join(OUTPUT_ROOT, 'plots', 'angle')
    os.makedirs(out, exist_ok=True)
    e = errors(df, best, fits[best]['yaw_deg'], fits[best]['mirror'])
    for rec, g in df.groupby('rec'):
        path = os.path.join(out, rec.replace('/', '_') + '_angle.png')
        figure(g, e.loc[g.index], path, f'{rec}: angle check, {best}, yaw {fits[best]["yaw_deg"]} deg, '
                                        f'mirror {fits[best]["mirror"]}')
        print(path)
