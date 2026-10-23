"""Diagnostic timeline: raw radar maps next to MoCap, to check alignment, geometry and features."""
import matplotlib.pyplot as plt
import numpy as np

from ..config import FPS, RANGE_AXIS, VELOCITY_AXIS
from ..groundtruth import active_speed_thresholds
from ..radar.cube import load_cube
from ..radar.features import MOVING, noise_floor
from ..utils import log_level, nan_smooth


def _maps(cube, step=5, chunk=2000):
    """moving power above the floor as range-time and Doppler-time maps (dB), max over `step` frames"""
    floor = noise_floor(cube)
    T = len(cube) // step * step
    rng = np.empty((T // step, cube.shape[1]), np.float32)
    dop = np.empty((T // step, cube.shape[2]), np.float32)
    for s in range(0, T, chunk):
        db = np.asarray(cube[s:min(s + chunk, T)], np.float32) - floor
        p = (10 ** (np.clip(db, 0, None) / 10) - 1) * MOVING
        rng[s // step:(s + len(db)) // step] = (10 * np.log10(p.sum(2) + 1e-3)).reshape(-1, step, p.shape[1]).max(1)
        dop[s // step:(s + len(db)) // step] = (10 * np.log10(p.sum(1) + 1e-3)).reshape(-1, step, p.shape[2]).max(1)
    return rng, dop, T


def timeline(session, path, features=('p_sub', 'p_r4'), count_feature='n_cells'):
    """labels, MoCap speed and window, radar features, range-time and Doppler-time maps with MoCap overlays"""
    s = session
    tt = s.t - s.t[0]
    rng, dop, T = _maps(load_cube(s.person, s.scenario))
    fig, ax = plt.subplots(4, 1, figsize=(22, 15), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1.4, 1.4]))
    for i, r in s.labels.iterrows():
        for a in ax[:2]:
            a.axvspan(r.start - s.t[0], r.end - s.t[0], color=f'C{i % 10}', alpha=0.15)
        ax[0].text((r.start + r.end) / 2 - s.t[0], 1.0, str(i), ha='center', fontsize=7,
                   transform=ax[0].get_xaxis_transform())
    if s.mocap:
        ax[0].plot(tt, nan_smooth(np.sqrt(s.mocap['ke']), 10), 'k', lw=0.8, label='MoCap RMS joint speed (m/s)')
        ax[0].plot(tt, nan_smooth(np.sqrt(s.mocap['vr2']), 10), 'r', lw=0.8, label='MoCap RMS radial joint speed')
        ax[2].plot(tt, s.mocap['dist_cog'], 'c', lw=0.8, label='MoCap distance radar - CoG (m)')
        ax[3].plot(tt, s.mocap['vr_cog'], 'c', lw=0.6, label='MoCap CoG radial speed (+ = away)')
    ax[0].fill_between(tt, 0, 1, where=s.masks['has_mocap'] & ~s.masks['window'], color='grey', alpha=0.4,
                       transform=ax[0].get_xaxis_transform(), label='MoCap, outside the evaluation window')
    try:
        ax[0].axhline(active_speed_thresholds()[s.person], color='k', ls=':', lw=0.8, label='MoCap active threshold')
    except (FileNotFoundError, KeyError):
        pass
    ax[0].set_ylim(0, 1.5)
    ax[0].set_title(f'{s.name}; offset {s.alignment["offset_s"]:+.2f} s after the logged capture start; '
                    'numbers = label rows')
    for name, c in zip(features, ('C0', 'C1')):
        ax[1].plot(tt, nan_smooth(log_level(s.radar[name]), 10), c, lw=0.8, label=f'log level {name}')
    ax2 = ax[1].twinx()
    ax2.plot(tt, nan_smooth(s.radar[count_feature], 10), 'C2', lw=0.8, label=count_feature)
    ax2.legend(loc='upper right', fontsize=8)
    ax[2].imshow(rng.T, aspect='auto', origin='lower', extent=[0, T / FPS, RANGE_AXIS[0], RANGE_AXIS[-1]],
                 cmap='magma', vmin=3, vmax=30)
    ax[2].set_ylim(0, RANGE_AXIS[-1])
    ax[2].set_ylabel('range (m)')
    ax[3].imshow(dop.T, aspect='auto', origin='lower', extent=[0, T / FPS, VELOCITY_AXIS[0], VELOCITY_AXIS[-1]],
                 cmap='magma', vmin=3, vmax=30)
    ax[3].set_ylim(-2, 2)
    ax[3].set_ylabel('radial velocity (m/s)')
    ax[3].set_xlabel('s since radar frame 0')
    for a in (ax[0], ax[1], ax[2], ax[3]):
        a.legend(loc='upper left' if a is ax[1] else 'upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)
