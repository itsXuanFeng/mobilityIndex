"""
Radar vs MoCap over time: how well the two sensors match, epoch by epoch.

recording_figure   the whole recording: label classes, movement of both sensors, agreement of the active calls,
                   distance to the radar
episodes_figure    close-ups of typical episodes (chair, bed, floor) at frame resolution, with the label names

Movement is shown as the z-score of log energy (radar: the person's movement energy e_move, from the point cloud;
MoCap: whole-body mean squared joint speed). The two have different units, so each is standardised over the evaluated
epochs of the recording and they share one axis without a second scale.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..config import COVERAGE_RADIUS, FPS
from ..utils import log_level, nan_smooth
from . import style
from .common import EPISODES, episode_span, hatch_unevaluated, label_spans, strip_axes

SERIES = [plt.Line2D([], [], color=style.MOCAP, label='MoCap: whole-body movement'),
          plt.Line2D([], [], color=style.RADAR, label='Radar: movement of the person (point cloud)')]

LEGEND = [  # the "active?" row
    Patch(facecolor=style.AGREE_ACTIVE, label='both active'),
    Patch(facecolor=style.AGREE_INACTIVE, label='both inactive'),
    Patch(facecolor=style.ONLY_MOCAP, label='only MoCap active (radar missed)'),
    Patch(facecolor=style.ONLY_RADAR, label='only radar active'),
    Patch(facecolor=style.SURFACE, edgecolor=style.HATCH, hatch='////', label='not evaluated'),
]


def _z(x, ref):
    """z-score of the log level of x, with mean and SD over the reference samples"""
    y = log_level(x)
    return (y - np.nanmean(y[ref])) / np.nanstd(y[ref])


def _agreement_colours(p):
    c = np.full(len(p), None, object)
    r, m = p.radar_active.values, p.mocap_active.values
    c[r & m] = style.AGREE_ACTIVE
    c[~r & ~m] = style.AGREE_INACTIVE
    c[m & ~r] = style.ONLY_MOCAP
    c[r & ~m] = style.ONLY_RADAR
    c[~p.evaluated.values] = None
    return c


def _strip(ax, x, width, colours, label):
    """a row of cells starting at x, None = hatched (not evaluated / no class)"""
    for xi, ci in zip(x, colours):
        if ci is None:
            ax.add_patch(plt.Rectangle((xi, 0), width, 1, facecolor=style.SURFACE, edgecolor=style.HATCH,
                                       hatch='////', linewidth=0))
        else:
            ax.add_patch(plt.Rectangle((xi, 0), width, 1, facecolor=ci, linewidth=0))
    strip_axes(ax, label)


def recording_figure(session, pairs, path):
    """the whole recording; pairs from mobility.evaluation.epoch_pairs"""
    style.apply()
    t0 = session.t[0]
    x = (pairs.t.values - t0) / 60                      # minutes since radar start
    w = 1 / 60                                           # one epoch
    ev = pairs.evaluated.values
    fig, ax = plt.subplots(4, 1, figsize=(12, 6.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.32, 2.4, 0.32, 1.0], hspace=0.18))
    label_spans(ax[0], session.labels, t0, 60)
    strip_axes(ax[0], 'activity\nlabel')

    zr = _z(pairs.radar_energy.values, ev)
    zm = _z(pairs.mocap_energy.values, ev)
    hatch_unevaluated(ax[1], x, w, ev)
    ax[1].plot(x, zm, color=style.MOCAP, label='MoCap: whole-body movement')
    ax[1].plot(x, zr, color=style.RADAR, label='Radar: movement of the person (point cloud)')
    ax[1].set_ylabel('movement\n(z-score of log energy)')

    _strip(ax[2], x, w, _agreement_colours(pairs), 'active?')

    ax[3].plot(x, pairs.dist.values, color=style.INK_2, linewidth=1.2)
    ax[3].axhline(COVERAGE_RADIUS, color=style.BASELINE, linewidth=0.8)
    ax[3].text(x[-1], COVERAGE_RADIUS, f'{COVERAGE_RADIUS:.0f} m: edge of radar coverage used ', ha='right',
               va='bottom', fontsize=7.5, color=style.MUTED)
    ax[3].set_ylabel('subject distance\nto radar (m)')
    ax[3].set_ylim(0, 8)
    ax[3].set_xlabel('minutes since radar start')
    ax[3].set_xlim(x[0], x[-1] + w)

    act_r = (pairs.radar_active & pairs.evaluated).sum() / 60
    act_m = (pairs.mocap_active & pairs.evaluated).sum() / 60
    fig.suptitle(f'{session.person} / {session.scenario}: radar and MoCap over the recording', x=0.06, y=0.995,
                 ha='left', fontsize=12, fontweight='bold')
    fig.text(0.06, 0.945, f'Evaluated: {ev.sum() / 60:.1f} min (hatched: no MoCap, outside radar coverage, operator '
             f'or caregiver in the room). Active time: MoCap {act_m:.2f} min, radar {act_r:.2f} min.',
             fontsize=9, color=style.INK_2)
    fig.legend(handles=SERIES + LEGEND, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.16)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def episodes_figure(session, pairs, path, episodes=EPISODES):
    """close-ups of typical episodes at frame resolution; the z-scores use the same reference as recording_figure"""
    style.apply()
    spans = [(title, episode_span(session.labels, a, b)) for title, a, b in episodes]
    spans = [(title, s) for title, s in spans if s is not None]
    win = session.masks['window']
    zr = nan_smooth(_z(session.radar['e_move'], win), FPS // 2)
    zm = nan_smooth(_z(session.mocap['ke'], win), FPS // 2)
    widths = [s[1] - s[0] for _, s in spans]
    fig, ax = plt.subplots(3, len(spans), figsize=(13, 5.6), sharey='row', squeeze=False,
                           gridspec_kw=dict(width_ratios=widths, height_ratios=[0.32, 2.4, 0.32], hspace=0.12,
                                            wspace=0.06))
    for c, (title, (a, b)) in enumerate(spans):
        f = (session.t >= a) & (session.t <= b)
        xs = session.t[f] - a
        e = (pairs.t.values >= a) & (pairs.t.values + 1 <= b)
        xe = pairs.t.values[e] - a
        label_spans(ax[0, c], session.labels, a, 1, names_until=b - a)
        ax[0, c].set_xlim(0, b - a)
        strip_axes(ax[0, c], 'activity\nlabel' if c == 0 else '')
        ax[0, c].set_title(title, pad=58)
        hatch_unevaluated(ax[1, c], xe, 1, pairs.evaluated.values[e])
        ax[1, c].plot(xs, zm[f], color=style.MOCAP, label='MoCap: whole-body movement')
        ax[1, c].plot(xs, zr[f], color=style.RADAR, label='Radar: movement of the person (point cloud)')
        ax[1, c].set_xlim(0, b - a)
        _strip(ax[2, c], xe, 1, _agreement_colours(pairs[e]), 'active?' if c == 0 else '')
        ax[2, c].set_xlim(0, b - a)
        ax[2, c].set_xlabel('seconds')
        for r in (0, 1):
            ax[r, c].tick_params(labelbottom=False)
        if c:
            ax[1, c].tick_params(labelleft=False)
    ax[1, 0].set_ylabel('movement\n(z-score of log energy, 0.5 s)')
    fig.suptitle(f'{session.person} / {session.scenario}: typical episodes, close up', x=0.06, y=0.995, ha='left',
                 fontsize=12, fontweight='bold')
    fig.legend(handles=SERIES + LEGEND, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.06, 0.945, 'Label colours: dark = active, light = inactive (arm-only tasks), hatched = not used. '
             'Same standardisation as the whole-recording figure.', fontsize=9, color=style.INK_2)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.76, bottom=0.17)
    fig.savefig(path, dpi=110)
    plt.close(fig)
