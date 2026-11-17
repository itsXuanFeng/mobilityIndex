"""
One recording, ready for metrics: per radar frame, the radar features, the aligned MoCap signals, the labels and the
masks of the evaluation window. Built from the cube (mobility.radar.cube); cached in OUTPUT_ROOT/<person>/<scenario>/:
    features.npz, angle_features.npz    radar features (mobility.radar.cache)
    subject.json, mocap_50hz.npz        MoCap shot, subject track and its signals (mobility.groundtruth)
    alignment.json                      time offset of radar frame 0; recomputed when the cube is newer

Evaluation window ('window' mask): MoCap present, subject within radar coverage, inside the scripted period (no
operator setting up or clearing the room), and outside the caregiver visit.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .align import fine_offset
from .config import COVERAGE_RADIUS, EPOCH_FRAMES, FPS, ROOM_MAX_X, recording_dir
from .groundtruth import mocap_50hz, subject, to_frames
from .io.recordings import capture_times, labels
from .labels import caregiver_period, script_period
from .radar.cache import angle_features, radar_features
from .radar.cube import newer_than_cube
from .utils import epoch_mean


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

    def label_per_epoch(self, n=EPOCH_FRAMES):
        """the scripted activity at the middle of each epoch of n frames ('' between labels)"""
        return self.label_per_frame()[n // 2::n]

    def epoch_mask(self, name, n=EPOCH_FRAMES):
        """(E,) bool per epoch of n frames: at least half of its frames are in the mask `name`"""
        return epoch_mean(self.masks[name], n) >= 0.5


def _alignment(person, scenario, overwrite):
    path = os.path.join(recording_dir(person, scenario), 'alignment.json')
    if newer_than_cube(path, person, scenario) and not overwrite:
        with open(path) as fp:
            al = json.load(fp)
        if 'shot' not in al:          # older caches also held the MoCap shot and track
            return al
    al = dict(capture_start=capture_times(person, scenario)[0], offset_s=0.0)
    sig = mocap_50hz(person, scenario, overwrite)
    if sig is not None:
        fit = fine_offset(radar_features(person, scenario, False)['p_sub'], al['capture_start'], sig)
        al.update({k: v for k, v in fit.items() if k != 'curve'})
    with open(path, 'w') as fp:
        json.dump(al, fp, indent=1)
    return al


def load_session(person, scenario, overwrite=False):
    al = _alignment(person, scenario, overwrite)
    al.update(subject(person, scenario))
    radar = radar_features(person, scenario, overwrite)
    T = len(next(iter(radar.values())))
    radar.update(angle_features(person, scenario, T, overwrite))
    t = al['capture_start'] + al['offset_s'] + np.arange(T) / FPS
    lb = labels(person, scenario)
    sig = mocap_50hz(person, scenario)
    gt = to_frames(sig, t) if sig is not None else {}
    has_mocap = ~np.isnan(gt['ke']) if gt else np.zeros(T, bool)
    covered = (gt['dist_cog'] < COVERAGE_RADIUS) & (gt['cog_x'] < ROOM_MAX_X) if gt else np.zeros(T, bool)
    s0, s1 = script_period(lb)
    in_script = (t >= s0) & (t <= s1)
    care = caregiver_period(lb)
    no_caregiver = ~((t >= care[0]) & (t <= care[1])) if care else np.ones(T, bool)
    no_data = np.isnan(radar['p_sub'])
    for k in ('e_body', 'v_horiz', 'e_move'):         # 0 means "nothing seen moving", not "no data"
        if k in radar:
            radar[k] = np.where(no_data, np.nan, radar[k])
    masks = dict(has_mocap=has_mocap, covered=covered, in_script=in_script, no_caregiver=no_caregiver,
                 window=has_mocap & covered & in_script & no_caregiver & ~no_data)
    return Session(person, scenario, t, radar, gt, lb, masks, al)
