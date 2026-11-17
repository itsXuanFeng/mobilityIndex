"""The angle checked against MoCap (scripts/fit_radar_orientation.py): radar positions vs the MoCap joint over time."""
import matplotlib.pyplot as plt

from ..config import RADAR_XYZ
from . import style


def angle_figure(df, err, path, title):
    """x, z and height over time of the radar's nearest moving reflector (from the angle) and of the MoCap joint
    nearest the radar. df: frames (t, r), err: their positions and errors, both from fit_radar_orientation.py"""
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
