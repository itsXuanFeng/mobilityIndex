"""
Fit the radar position in the MoCap frame: nearest moving range (feature r_near) vs the MoCap distance from the radar
to the nearest of head, shoulders, wrists, knees, over frames with clear movement. Put the result in
mobility/config.py RADAR_XYZ.

    python scripts/fit_radar_position.py --only person_1/scenario_1 person_5/scenario_3 person_10/scenario_2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.align import FIT_JOINTS, fit_radar_position  # noqa: E402
from mobility.config import RADAR_XYZ  # noqa: E402
from mobility.groundtruth import joint_kinematics  # noqa: E402
from mobility.io.mocap import load_track  # noqa: E402
from mobility.io.recordings import find_recordings, parse_recording  # noqa: E402
from mobility.radar.cube import cube_path  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--min-joint-speed', type=float, default=0.3, help='m/s, fastest joint, to use a frame')
    args = ap.parse_args()
    recs = [parse_recording(r) for r in args.only] if args.only else find_recordings()
    ranges, joints = [], []
    for p, s in recs:
        if not os.path.exists(cube_path(p, s)):
            continue
        ses = load_session(p, s)
        t, pos = load_track(ses.alignment['track'], FIT_JOINTS)
        grid, pj, v = joint_kinematics(t, pos)
        k = np.clip(np.searchsorted(grid, ses.t), 0, len(grid) - 1)
        speed = np.linalg.norm(v[k], axis=-1).max(1)
        use = ses.masks['has_mocap'] & ~np.isnan(ses.radar['r_near']) & (speed > args.min_joint_speed)
        ranges.append(ses.radar['r_near'][use])
        joints.append(pj[k[use]])
        print(f'{ses.name}: {use.sum()} frames')
    ranges, joints = np.concatenate(ranges), np.concatenate(joints)
    fit = fit_radar_position(ranges, joints)
    now = np.median(np.abs(np.linalg.norm(joints - np.asarray(RADAR_XYZ), axis=-1).min(1) - ranges))
    print(f'fitted RADAR_XYZ = ({fit["xyz"][0]:.2f}, {fit["xyz"][1]:.2f}, {fit["xyz"][2]:.2f}): median |range error| '
          f'{100 * fit["median_abs_err"]:.1f} cm, bias {100 * fit["bias"]:+.1f} cm, {len(ranges)} frames')
    print(f'current RADAR_XYZ = {RADAR_XYZ}: median |range error| {100 * now:.1f} cm')
