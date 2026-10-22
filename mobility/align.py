"""
Radar <-> MoCap alignment.

Time: radar frame j is at capture_start + offset + j / FPS. The capture start is logged to the whole second only, so the
offset (found 0.2-0.95 s) is estimated per recording by maximising the correlation between radar moving power and
MoCap kinetic energy. The peak is sharp: 1 s off already destroys the agreement.

Space: the radar position in the MoCap frame is fitted from the range of the nearest moving reflector vs the MoCap
distance from the radar to the nearest upper-body / knee joint.
"""
import numpy as np

from .config import ALIGN_SEARCH_S, FPS
from .utils import log_level, nan_smooth

FIT_JOINTS = ['HeadPosition', 'LShoulderPositions', 'RShoulderPositions', 'LWristPositions', 'RWristPositions',
              'LKneePositions', 'RKneePositions']


def fine_offset(radar_power, capture_start, mocap, search=ALIGN_SEARCH_S, step=0.05):
    """
    :param radar_power: per-frame moving power (e.g. feature p_sub)
    :param mocap: 50 Hz MoCap signals (mobility.groundtruth.mocap_signals)
    :return: dict(offset_s, corr, margin = corr minus the best corr more than 2 s away, curve (offsets, corrs))
    """
    rx = nan_smooth(log_level(radar_power), FPS // 2)
    mk = np.log10(mocap['ke'] + 1e-3)
    frames = np.arange(len(rx)) / FPS
    offsets = np.arange(search[0], search[1] + 1e-9, step)
    corrs = np.full(len(offsets), np.nan)
    for i, off in enumerate(offsets):
        m = np.interp(capture_start + off + frames, mocap['t'], mk, left=np.nan, right=np.nan)
        ok = ~np.isnan(m) & ~np.isnan(rx)
        if ok.sum() > 10 * FPS:
            corrs[i] = np.corrcoef(rx[ok], m[ok])[0, 1]
    best = int(np.nanargmax(corrs))
    far = np.abs(offsets - offsets[best]) > 2
    return dict(offset_s=float(offsets[best]), corr=float(corrs[best]),
                margin=float(corrs[best] - np.nanmax(corrs[far])), curve=(offsets, corrs))


def fit_radar_position(ranges, joints, heights=np.arange(2.3, 3.31, 0.05), xs=np.arange(-4.25, -2.24, 0.1),
                       zs=np.arange(-0.5, 1.51, 0.1)):
    """
    grid search of the radar position minimising the median |nearest-joint distance - measured range|
    :param ranges: (N,) range of the nearest moving reflector (feature r_near), frames with clear movement
    :param joints: (N, J, 3) MoCap positions of FIT_JOINTS at those frames
    :return: dict(xyz, median_abs_err, bias)
    """
    best = None
    for h in heights:
        for x in xs:
            for z in zs:
                d = np.linalg.norm(joints - [x, h, z], axis=-1).min(1)
                err = np.median(np.abs(d - ranges))
                if best is None or err < best[0]:
                    best = (err, (float(x), float(h), float(z)))
    err, xyz = best
    d = np.linalg.norm(joints - xyz, axis=-1).min(1)
    return dict(xyz=xyz, median_abs_err=float(err), bias=float(np.median(ranges - d)))
