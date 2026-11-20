"""
Per-recording caches of the per-frame radar features, in OUTPUT_ROOT/cache/<person>/<scenario>/:
    features.npz          radar features of the cube (mobility.radar.features) and the noise floor `_floor_db`;
                          missing ones are added when new features are registered; recomputed when the cube is newer
    angle_features.npz    the person's height, position and speeds per frame, from the angle pass
                          (mobility.radar.angle.frame_features; only when points.npz exists); recomputed when the
                          points are newer
"""
import os

import numpy as np

from ..config import recording_dir
from .angle import ANGLE_FEATURES, frame_features
from .cube import load_cube, newer_than_cube
from .features import FEATURES, compute_features


def features_path(person, scenario):
    return os.path.join(recording_dir(person, scenario), 'features.npz')


def radar_features(person, scenario, overwrite=False):
    """{name: (T,)} of every registered feature, from the cache, computing (and caching) the missing ones"""
    path = features_path(person, scenario)
    have = {}
    if newer_than_cube(path, person, scenario) and not overwrite:
        have = dict(np.load(path))
    missing = [n for n in FEATURES if n not in have]
    if missing:
        new, floor = compute_features(load_cube(person, scenario), missing, floor_db=have.get('_floor_db'))
        have.update(new)
        have['_floor_db'] = floor
        np.savez(path, **have)
    return {k: have[k] for k in FEATURES}             # caches may still hold features that were removed


def recording_noise_floor(person, scenario):
    """(range, Doppler) noise floor of the recording in dB, the one its features use (computed from the cube)"""
    radar_features(person, scenario, False)
    return np.load(features_path(person, scenario))['_floor_db']


def angle_features(person, scenario, n_frames, overwrite=False):
    """per-frame features of the person from the angle-pass points (mobility.radar.angle.frame_features), cached in
    angle_features.npz; {} when the recording has no points.npz (scripts/angle_pass.py)"""
    from .points import load_points, points_path    # points.py needs the noise floor from here
    src = points_path(person, scenario)
    if not os.path.exists(src):
        return {}
    path = os.path.join(recording_dir(person, scenario), 'angle_features.npz')
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(src) and not overwrite:
        have = dict(np.load(path))
        if set(ANGLE_FEATURES) <= set(have) and all(len(v) == n_frames for v in have.values()):
            return have
    have = frame_features(load_points(person, scenario), recording_noise_floor(person, scenario), n_frames)
    np.savez(path, **have)
    return have
