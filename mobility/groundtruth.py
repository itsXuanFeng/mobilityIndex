"""
MoCap ground truth.

- Per 50 Hz sample: joint speeds and derived signals of the subject's skeleton (mocap_signals), cached per recording.
- Per 1 s epoch on the MoCap clock: RMS joint speed and the activity-label class (mocap_epochs). No radar needed, so
  every recording with MoCap and labels can be used.
- The active threshold on RMS joint speed: calibrated against the label classes, leaving the evaluated person out
  (calibrate_active_speed, scripts/calibrate_mocap_threshold.py). Nothing is set by hand.

To add a ground-truth signal, add it to the dict in mocap_signals and its name to SIGNALS; it is carried to radar frames
by to_frames automatically, and cached files without it are recomputed.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import EPOCH_S, GROUNDTRUTH_DIR, RADAR_XYZ, recording_dir
from .io.mocap import JOINTS, MOCAP_FPS, load_track, shot_for, subject_track
from .io.recordings import labels as load_labels
from .labels import ACTIVE, INACTIVE, time_classes
from .utils import best_threshold

THRESHOLD_FILE = os.path.join(GROUNDTRUTH_DIR, 'mocap_active_threshold.csv')
SIGNALS = ('ke', 'vr2', 'vr_cog', 'vr_max', 'dist_cog', 'cog_x', 'cog_z', 'head_y', 'hip_y', 'knee_y', 'trunk_deg', 'knee_deg')


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
        ke         mean squared joint speed (m^2/s^2): whole-body kinetic-energy proxy
        vr2        mean squared joint speed along the line to the radar: what the radar can see
        vr_cog     signed radial speed of the centre of gravity (+ = away from the radar)
        vr_max     largest |radial speed| over the joints (m/s): what the peak Doppler sees (Category 3)
        dist_cog   distance radar - centre of gravity (m)
        cog_x, cog_z, head_y   position (m)
        hip_y      height of the pelvis (mean of the two hips, m)                    posture (Category 3)
        knee_y     height of the knees (mean of both, m): near the floor when kneeling
        trunk_deg  angle of the trunk (hips -> neck) from vertical: 0 upright, 90 lying
        knee_deg   knee angle (hip-knee-ankle, mean of both sides): 180 straight
    """
    t, pos = load_track(track_csv)
    grid, p, v = joint_kinematics(t, pos)
    u = p - np.asarray(radar_xyz)
    dist = np.linalg.norm(u, axis=-1)
    vr = (v * u).sum(-1) / dist
    j = {name: p[:, i] for i, name in enumerate(JOINTS)}
    hip = (j['LHipPositions'] + j['RHipPositions']) / 2
    trunk = j['NeckPosition'] - hip
    return dict(t=grid, ke=(v ** 2).sum(-1).mean(-1), vr2=(vr ** 2).mean(-1), vr_cog=vr[:, 0],
                vr_max=np.abs(vr).max(-1), dist_cog=dist[:, 0], cog_x=p[:, 0, 0], cog_z=p[:, 0, 2],
                head_y=j['HeadPosition'][:, 1], hip_y=hip[:, 1],
                knee_y=(j['LKneePositions'][:, 1] + j['RKneePositions'][:, 1]) / 2,
                trunk_deg=np.degrees(np.arccos(np.clip(trunk[:, 1] / np.linalg.norm(trunk, axis=-1), -1, 1))),
                knee_deg=(_angle(j['LHipPositions'], j['LKneePositions'], j['LAnklePositions'])
                          + _angle(j['RHipPositions'], j['RKneePositions'], j['RAnklePositions'])) / 2)


def _angle(a, b, c):
    """angle (degrees) at b between b->a and b->c"""
    u, w = a - b, c - b
    cos = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def subject(person, scenario, overwrite=False):
    """the recording's MoCap shot and subject track, {'shot', 'track'} (None if no shot); cached in subject.json"""
    path = os.path.join(recording_dir(person, scenario), 'subject.json')
    if os.path.exists(path) and not overwrite:
        with open(path) as fp:
            return json.load(fp)
    shot = shot_for(person, scenario)
    sub = dict(shot=shot['shot'], track=subject_track(shot)) if shot else dict(shot=None, track=None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fp:
        json.dump(sub, fp, indent=1)
    return sub


def mocap_50hz(person, scenario, overwrite=False):
    """mocap_signals of the subject's track, or None without MoCap; cached in mocap_50hz.npz (recomputed when the
    track or RADAR_XYZ change, or a signal in SIGNALS is missing)"""
    track = subject(person, scenario, overwrite)['track']
    if track is None:
        return None
    path = os.path.join(recording_dir(person, scenario), 'mocap_50hz.npz')
    if os.path.exists(path) and not overwrite:
        d = dict(np.load(path))
        if tuple(d.pop('_radar_xyz')) == tuple(RADAR_XYZ) and str(d.pop('_track')) == track and set(SIGNALS) <= set(d):
            return d
    d = mocap_signals(track, RADAR_XYZ)
    np.savez(path, _radar_xyz=np.array(RADAR_XYZ), _track=track, **d)
    return d


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


def mocap_epochs(person, scenario):
    """1 s epochs on the MoCap clock: start time, RMS joint speed (m/s) and label class (active / inactive / None)"""
    sig = mocap_50hz(person, scenario)
    if sig is None:
        return None
    n = int(round(EPOCH_S * MOCAP_FPS))
    E = len(sig['t']) // n
    t = sig['t'][:E * n].reshape(E, n)[:, 0]
    speed = np.sqrt(sig['ke'][:E * n].reshape(E, n).mean(1))
    cls = time_classes(t + EPOCH_S / 2, load_labels(person, scenario))
    return pd.DataFrame(dict(person=person, scenario=scenario, t=t, speed=speed, cls=cls))


def calibrate_active_speed(epochs, exclude_person=None):
    """RMS joint-speed threshold that best separates active from inactive label epochs (balanced accuracy), on all
    persons except exclude_person"""
    e = epochs[epochs.cls.isin([ACTIVE, INACTIVE]) & (epochs.person != exclude_person)]
    return best_threshold(e.speed.values, (e.cls == ACTIVE).values)


def active_speed_thresholds(path=THRESHOLD_FILE):
    """{person: threshold calibrated without that person, 'ALL': on every person}, from
    scripts/calibrate_mocap_threshold.py"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run scripts/calibrate_mocap_threshold.py first')
    d = pd.read_csv(path)
    return dict(zip(d.person, d.threshold))
