"""
MoCap ground truth: per 50 Hz sample signals from the subject's skeleton, and their resampling to radar frames.

To add a ground-truth signal (e.g. trunk angle for posture, Category 3), add it to the dict in mocap_signals; it is
carried to radar frames by to_frames automatically.
"""
import numpy as np
from scipy.signal import savgol_filter

from .config import RADAR_XYZ
from .io.mocap import JOINTS, MOCAP_FPS, load_track


def joint_kinematics(t, pos):
    """positions and velocities (m/s) on a uniform 50 Hz grid; Savitzky-Golay (13 samples = 0.26 s, cubic)"""
    dt = 1 / MOCAP_FPS
    grid = np.arange(t[0], t[-1], dt)
    p = np.stack([np.stack([np.interp(grid, t, pos[:, j, k]) for k in range(3)], -1)
                  for j in range(pos.shape[1])], 1)
    v = savgol_filter(p, 13, 3, deriv=1, delta=dt, axis=0)
    p = savgol_filter(p, 13, 3, axis=0)
    return grid, p, v


def mocap_signals(track_csv, radar_xyz=RADAR_XYZ):
    """
    per 50 Hz sample:
        t          epoch s
        ke         mean squared joint speed (m^2/s^2): kinetic-energy proxy
        vr2        mean squared joint speed along the line to the radar: what the radar can see
        vr_cog     signed radial speed of the centre of gravity (+ = away from the radar)
        dist_cog   distance radar - centre of gravity (m)
        cog_x, cog_z, head_y   position (m)
    """
    t, pos = load_track(track_csv)
    grid, p, v = joint_kinematics(t, pos)
    u = p - np.asarray(radar_xyz)
    dist = np.linalg.norm(u, axis=-1)
    vr = (v * u).sum(-1) / dist
    return dict(t=grid, ke=(v ** 2).sum(-1).mean(-1), vr2=(vr ** 2).mean(-1), vr_cog=vr[:, 0], dist_cog=dist[:, 0],
                cog_x=p[:, 0, 0], cog_z=p[:, 0, 2], head_y=p[:, JOINTS.index('HeadPosition'), 1])


def to_frames(sig, t_frames, frame_s=0.05):
    """mean of every signal over each radar frame (frame_s around its time); NaN where MoCap doesn't cover it"""
    t = sig['t']
    edges = np.concatenate([t_frames - frame_s / 2, [t_frames[-1] + frame_s / 2]])
    idx = np.searchsorted(t, edges)
    n = np.maximum(idx[1:] - idx[:-1], 1)
    covered = (t_frames >= t[0]) & (t_frames <= t[-1])
    out = {}
    for k, x in sig.items():
        if k == 't':
            continue
        c = np.concatenate([[0], np.cumsum(x)])
        y = (c[idx[1:]] - c[idx[:-1]]) / n
        y[~covered] = np.nan
        out[k] = y
    return out
