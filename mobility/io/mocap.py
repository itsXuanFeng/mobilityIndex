"""
MoCap (Captury) shots: pairing them with radar recordings, picking the subject's track, loading joint positions.

A shot folder holds one .csv per track. Tracks include ghosts, so the subject is chosen by movement, not file size.
Shots are paired with radar recordings by time overlap, not by number (e.g. person_1 recorded scenario_4 before 3).
"""
import glob
import os

import numpy as np
import pandas as pd

from ..config import MOCAP_ROOT, ROOM_MAX_X
from .recordings import capture_times

MOCAP_FPS = 50
JOINTS = ['CenterOfGravity', 'HeadPosition', 'NeckPosition', 'SpinePosition', 'LWristPositions', 'RWristPositions',
          'LElbowPositions', 'RElbowPositions', 'LAnklePositions', 'RAnklePositions', 'LKneePositions',
          'RKneePositions', 'LHipPositions', 'RHipPositions', 'LShoulderPositions', 'RShoulderPositions']


def shots(person):
    """per shot: name, start / end (epoch s) and its track .csv files, largest first"""
    out = []
    for d in sorted(glob.glob(f'{MOCAP_ROOT}/{person}/shot_*')):
        with open(f'{d}/meta.txt') as fp:
            meta = dict(line.rstrip('\n').split('\t', 1) for line in fp if '\t' in line)
        start = int(meta['StartTimestamp']) / 1e6
        tracks = [f for f in glob.glob(f'{d}/*.csv') if not f.endswith(('.face.csv', '.landmarks.csv'))]
        tracks.sort(key=os.path.getsize, reverse=True)
        out.append(dict(shot=os.path.basename(d), start=start,
                        end=start + int(meta['DurationInFrames']) / MOCAP_FPS, tracks=tracks))
    return out


def shot_for(person, scenario):
    """the shot that overlaps the radar recording most, or None"""
    r0, r1 = capture_times(person, scenario)
    candidates = [s for s in shots(person) if s['tracks']]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: min(r1, s['end']) - max(r0, s['start']))
    return best if min(r1, best['end']) - max(r0, best['start']) > 0 else None


def load_track(csv, joints=JOINTS):
    """timestamps (epoch s) and joint positions (T, J, 3) in metres, MoCap frame (Y up).
    Header: line 2 holds the joint names (one per X/Y/Z(/W) group), data from line 5; a footer holds camera info."""
    with open(csv) as fp:
        head = [next(fp) for _ in range(5)]
    cols, cur = {}, ''
    for i, name in enumerate(head[2].rstrip('\n').split(';')):
        cur = name or cur
        cols.setdefault(cur, i)
    use = [2] + [cols[j] + k for j in joints for k in range(3)]
    df = pd.read_csv(csv, sep=';', skiprows=5, header=None, usecols=use, low_memory=False)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[(df[2] > 1.7e15) & df[use[1:]].notna().all(axis=1)]   # drops the footer
    t = df[2].to_numpy() / 1e6
    pos = df[use[1:]].to_numpy(float).reshape(len(df), len(joints), 3) / 1000
    return t, pos


def subject_track(shot, max_tracks=5):
    """the subject's track: the longest centre-of-gravity path inside the room. The largest file is not always the
    subject (person_4 shot_006, person_5 shot_005 and shot_006: static ghost tracks)."""
    best, best_len = None, -1.0
    for f in shot['tracks'][:max_tracks]:
        _, pos = load_track(f, joints=['CenterOfGravity'])
        c = pos[:, 0]
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        inside = (c[1:, 0] < ROOM_MAX_X) & (c[:-1, 0] < ROOM_MAX_X)
        length = step[inside & (step < 0.1)].sum()   # 0.1 m per 20 ms: tracking jumps aren't movement
        if length > best_len:
            best, best_len = f, length
    return best
