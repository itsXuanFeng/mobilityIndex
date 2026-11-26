"""
Activity bouts of the radar and of MoCap over time (Category 2): the whole recording, and a close-up of the bed
episode, where most bouts disagree.

Per sensor, one row: its bouts as bars (solid = matched to a bout of the other sensor, outline = unmatched) above its
active epochs (grey ticks: above the entry threshold, before the hysteresis, the confirmation and the bout rule), and
its posture transitions (black triangles), which confirm a bout too.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..bouts import match_bouts
from ..utils import runs
from . import style
from .common import EPISODES, episode_span, hatch_unevaluated, label_spans, strip_axes

LEGEND = [
    Patch(facecolor=style.MOCAP, label='MoCap bout, matched'),
    Patch(facecolor=style.RADAR, label='radar bout, matched'),
    Patch(facecolor=style.SURFACE, edgecolor=style.INK_2, label='bout without a match (outline)'),
    Patch(facecolor=style.ACTIVE, label='active epoch (entry), before hysteresis, confirmation, rule'),
    plt.Line2D([], [], color=style.INK, marker='v', linestyle='', markersize=5, label='posture transition'),
    Patch(facecolor=style.SURFACE, edgecolor=style.HATCH, hatch='////', label='not evaluated'),
]


def _bout_row(ax, x, width, active, bouts, matched, colour, label, transitions=None):
    for i in np.flatnonzero(active):
        ax.add_patch(plt.Rectangle((x[i], 0.05), width, 0.25, facecolor=style.ACTIVE, linewidth=0))
    if transitions is not None and transitions.any():
        i = np.flatnonzero(transitions)
        ax.plot(x[i] + width / 2, np.full(len(i), 1.05), 'v', color=style.INK, markersize=4, clip_on=False)
    for k, (s, e) in enumerate(bouts):
        x0, w = x[s], x[e - 1] + width - x[s]
        if k in matched:
            ax.add_patch(plt.Rectangle((x0, 0.4), w, 0.55, facecolor=colour, linewidth=0))
        else:
            ax.add_patch(plt.Rectangle((x0, 0.4), w, 0.55, facecolor=style.SURFACE, edgecolor=colour, linewidth=1.2))
    strip_axes(ax, label)


def _rows(axes, session, pairs, t0, scale, names_until=None):
    """label row and the two bout rows over the epochs of pairs, x = (t - t0) / scale"""
    x = (pairs.t.values - t0) / scale
    w = 1 / scale
    mb, rb = runs(pairs.mocap_bout.values), runs(pairs.radar_bout.values)
    m = match_bouts(mb, rb)
    label_spans(axes[0], session.labels, t0, scale, names_until)
    strip_axes(axes[0], 'activity\nlabel')
    ev = pairs.evaluated.values
    for ax, src, b, matched, colour, label in ((axes[1], 'mocap', mb, set(m), style.MOCAP, 'MoCap'),
                                               (axes[2], 'radar', rb, set(m.values()), style.RADAR, 'radar')):
        hatch_unevaluated(ax, x, w, ev)
        tr = pairs[f'{src}_transition'].values & ev if f'{src}_transition' in pairs else None
        _bout_row(ax, x, w, pairs[f'{src}_active'].values & ev, b, matched, colour, label, tr)
    for ax in axes:
        ax.set_xlim(x[0], x[-1] + w)
    return len(mb), len(rb), len(m)


def bouts_figure(session, pairs, path, note='', episode=EPISODES[1]):
    """pairs from mobility.evaluation.bout_frame; note: the bout definition used, for the subtitle"""
    style.apply()
    span = episode_span(session.labels, episode[1], episode[2])
    fig, ax = plt.subplots(7 if span else 3, 1, figsize=(12, 6.4 if span else 3.2),
                           gridspec_kw=dict(height_ratios=[0.35, 0.6, 0.6] + ([1.15, 0.35, 0.6, 0.6] if span else []),
                                            hspace=0.15))
    t0 = session.t[0]
    n_m, n_r, n_matched = _rows(ax[:3], session, pairs, t0, 60)
    ax[2].set_xlabel('minutes since radar start')
    for a in ax[:2]:
        a.tick_params(labelbottom=False)
    if span:
        ax[3].set_visible(False)
        e = (pairs.t.values >= span[0]) & (pairs.t.values + 1 <= span[1])
        _rows(ax[4:], session, pairs[e], span[0], 1, names_until=span[1] - span[0])
        ax[4].set_title(f'{episode[0]}, close up', pad=58)
        ax[6].set_xlabel('seconds')
        for a in ax[4:6]:
            a.tick_params(labelbottom=False)
    fig.suptitle(f'{session.person} / {session.scenario}: activity bouts, radar and MoCap', x=0.06, y=0.995,
                 ha='left', fontsize=12, fontweight='bold')
    fig.text(0.06, 0.95, f'{note}Bouts: MoCap {n_m}, radar {n_r}, matched {n_matched}.', fontsize=9,
             color=style.INK_2)
    fig.legend(handles=LEGEND, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.14)
    fig.savefig(path, dpi=110)
    plt.close(fig)
