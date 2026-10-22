"""
One recording, ready for metrics: per radar frame, the radar features, the aligned MoCap signals, the labels and the
masks of the evaluation window. Built from the cube (mobility.radar.cube), cached in OUTPUT_ROOT/<person>/<scenario>/:
    features.npz          radar features (missing ones are added when new features are registered)
    mocap_50hz.npz        MoCap signals of the subject's track (recomputed when RADAR_XYZ changes)
    alignment.json        MoCap shot and track, time offset of radar frame 0
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .align import fine_offset
from .config import COVERAGE_RADIUS, FPS, RADAR_XYZ, ROOM_MAX_X
from .groundtruth import mocap_signals, to_frames
from .io.mocap import shot_for, subject_track
from .io.recordings import capture_times, labels
from .radar.cube import load_cube, recording_dir
from .radar.features import FEATURES, compute_features


@dataclass
class Session:
    person: str
    scenario: str
    t: np.ndarray            # (T,) time of each radar frame, epoch s (aligned to MoCap)
    radar: dict              # feature name -> (T,)
    mocap: dict              # MoCap signal -> (T,), NaN where MoCap doesn't cover the frame
    labels: pd.DataFrame
    masks: dict              # name -> (T,) bool; 'window' = frames used for evaluation
    alignment: dict

    @property
    def name(self):
        return f'{self.person}/{self.scenario}'

    def label_per_frame(self):
        """the scripted activity of each frame ('' between labels)"""
        lab = np.full(len(self.t), '', object)
        for _, r in self.labels.iterrows():
            lab[(self.t >= r.start) & (self.t <= r.end)] = r.label
        return lab


def _features(person, scenario, overwrite):
    path = os.path.join(recording_dir(person, scenario), 'features.npz')
    have = {}
    if os.path.exists(path) and not overwrite:
        have = dict(np.load(path))
    missing = [n for n in FEATURES if n not in have]
    if missing:
        new, floor = compute_features(load_cube(person, scenario), missing, floor_db=have.get('_floor_db'))
        have.update(new)
        have['_floor_db'] = floor
        np.savez(path, **have)
    return {k: v for k, v in have.items() if not k.startswith('_')}


def _mocap(person, scenario, track, overwrite):
    path = os.path.join(recording_dir(person, scenario), 'mocap_50hz.npz')
    if os.path.exists(path) and not overwrite:
        d = dict(np.load(path))
        if tuple(d.pop('_radar_xyz')) == tuple(RADAR_XYZ) and str(d.pop('_track')) == track:
            return d
    d = mocap_signals(track, RADAR_XYZ)
    np.savez(path, _radar_xyz=np.array(RADAR_XYZ), _track=track, **d)
    return d


def _alignment(person, scenario, overwrite):
    path = os.path.join(recording_dir(person, scenario), 'alignment.json')
    if os.path.exists(path) and not overwrite:
        with open(path) as fp:
            return json.load(fp)
    shot = shot_for(person, scenario)
    al = dict(capture_start=capture_times(person, scenario)[0], shot=None, track=None, offset_s=0.0)
    if shot is not None:
        al.update(shot=shot['shot'], track=subject_track(shot))
        power = _features(person, scenario, False)['p_sub']
        fit = fine_offset(power, al['capture_start'], _mocap(person, scenario, al['track'], False))
        al.update({k: v for k, v in fit.items() if k != 'curve'})
    with open(path, 'w') as fp:
        json.dump(al, fp, indent=1)
    return al


def load_session(person, scenario, overwrite=False):
    al = _alignment(person, scenario, overwrite)
    radar = _features(person, scenario, overwrite)
    T = len(next(iter(radar.values())))
    t = al['capture_start'] + al['offset_s'] + np.arange(T) / FPS
    lb = labels(person, scenario)
    if al['track']:
        gt = to_frames(_mocap(person, scenario, al['track'], overwrite), t)
    else:
        gt = {}
    has_mocap = ~np.isnan(gt['ke']) if gt else np.zeros(T, bool)
    covered = (gt['dist_cog'] < COVERAGE_RADIUS) & (gt['cog_x'] < ROOM_MAX_X) if gt else np.zeros(T, bool)
    care = lb[lb.label.str.contains('caregiver', case=False)]
    no_caregiver = ~((t >= care.start.min()) & (t <= care.end.max())) if len(care) else np.ones(T, bool)
    no_data = np.isnan(radar['p_sub'])
    masks = dict(has_mocap=has_mocap, covered=covered, no_caregiver=no_caregiver,
                 window=has_mocap & covered & no_caregiver & ~no_data)
    return Session(person, scenario, t, radar, gt, lb, masks, al)
