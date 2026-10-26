"""
Postural transitions of the radar and of MoCap over time (Category 3): the whole recording, and a close-up of the bed
episode.

Rows: the activity labels; the posture of both sensors (radar: from the height, carried forward while the person isn't
seen); the transitions of each sensor (down = away from the radar, up = towards it; filled = matched to a transition of
the same kind of the other sensor, outlined = not); and the height: the radar's top of the body (dots, from the angle,
also while the person is still) and the MoCap head.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..posture import LEVEL, POSTURES, UNKNOWN
from . import style
from .agreement import EPISODES, _episode_span, _hatch_unevaluated, _label_spans, _strip_axes

SHORT_KIND = {'sit_to_stand': 'stand up', 'stand_to_sit': 'sit down', 'lie_to_sit': 'sit up', 'sit_to_lie': 'lie down',
              'stand_to_lie': 'lie down', 'lie_to_stand': 'get up', 'to_floor': 'to floor', 'from_floor': 'from floor'}

LEGEND = [
    Line2D([], [], color=style.MOCAP, label='MoCap posture per frame (a change counts once held)'),
    Line2D([], [], color=style.RADAR, label='radar posture per frame, from the height'),
    Line2D([], [], marker='v', linestyle='', color=style.MOCAP, label='MoCap transition (down / up)'),
    Line2D([], [], marker='v', linestyle='', color=style.RADAR, label='radar transition'),
    Line2D([], [], marker='v', linestyle='', markerfacecolor=style.SURFACE, color=style.INK_2,
           label='without a match of the same kind (outline)'),
    Line2D([], [], marker='.', linestyle='', color=style.RADAR, label='radar: top of the body (angle)'),
    Line2D([], [], color=style.MOCAP, label='MoCap: head'),
    Patch(facecolor=style.SURFACE, edgecolor=style.HATCH, hatch='////', label='not evaluated'),
]


def _level(posture):
    return np.array([LEVEL.get(int(p), np.nan) if p != UNKNOWN else np.nan for p in posture])


def _rows(axes, session, posture, radar_posture, height, events, t0, scale, sel, names=False):
    """label, posture, transition and height rows over the frames sel, x = (t - t0) / scale"""
    t = session.t[sel]
    x = (t - t0) / scale
    w = 1 / (20 * scale)
    win = session.masks['window'][sel]
    _label_spans(axes[0], session.labels, t0, scale, names_until=(x[-1] - x[0]) if names else None)
    _strip_axes(axes[0], 'activity\nlabel')

    _hatch_unevaluated(axes[1], x, w, win)
    axes[1].plot(x, _level(posture[sel]) + 0.08, color=style.MOCAP, linewidth=1.2, drawstyle='steps-post')
    axes[1].plot(x, _level(radar_posture[sel]) - 0.08, color=style.RADAR, linewidth=1.2, drawstyle='steps-post')
    axes[1].set_yticks([LEVEL[k] for k in range(4)], [POSTURES[k] for k in range(4)])
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_ylabel('posture')

    _hatch_unevaluated(axes[2], x, w, win)
    when = events.t_s.values + session.t[0]                 # epoch s
    for src, y, colour in (('mocap', 0.7, style.MOCAP), ('radar', 0.3, style.RADAR)):
        e = events[(events.source == src).values & (when >= t[0]) & (when <= t[-1])]
        for r in e.itertuples():
            xe = (r.t_s + session.t[0] - t0) / scale
            axes[2].plot(xe, y, marker='v' if r.direction > 0 else '^', markersize=7, color=colour,
                         markerfacecolor=colour if r.matched else style.SURFACE, markeredgewidth=1.2)
            if names:
                axes[2].text(xe, y + 0.13, SHORT_KIND.get(r.kind, r.kind), ha='center', va='bottom', fontsize=6.5,
                             color=style.INK_2)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_yticks([0.3, 0.7], ['radar', 'MoCap'])
    axes[2].grid(False)
    axes[2].set_ylabel('transitions')

    _hatch_unevaluated(axes[3], x, w, win)
    axes[3].plot(x, session.mocap['head_y'][sel], color=style.MOCAP, linewidth=1.0)
    axes[3].plot(x, height[sel], '.', color=style.RADAR, markersize=1.5)
    axes[3].set_ylim(-0.1, 2.0)
    axes[3].set_ylabel('height (m)')
    for ax in axes:
        ax.set_xlim(x[0], x[-1] + w)


def transitions_figure(session, posture, radar_posture, height, events, path, episode=EPISODES[1]):
    """posture, radar_posture: posture code per frame of MoCap and of the radar; height: the radar's top-of-body
    height per frame; events: mobility.evaluation.transition_events of the recording (t in s since the first frame)"""
    style.apply()
    span = _episode_span(session.labels, episode[1], episode[2])
    ratios = [0.35, 0.9, 0.6, 1.2]
    fig, ax = plt.subplots(9 if span else 4, 1, figsize=(12, 10.5 if span else 5.2),
                           gridspec_kw=dict(height_ratios=ratios + ([0.9] + ratios if span else []), hspace=0.18))
    t0 = session.t[0]
    every = np.ones(len(session.t), bool)
    _rows(ax[:4], session, posture, radar_posture, height, events, t0, 60, every)
    ax[3].set_xlabel('minutes since radar start')
    for a in ax[:3]:
        a.tick_params(labelbottom=False)
    if span:
        ax[4].set_visible(False)
        sel = (session.t >= span[0]) & (session.t <= span[1])
        _rows(ax[5:], session, posture, radar_posture, height, events, span[0], 1, sel, names=True)
        ax[5].set_title(f'{episode[0]}, close up', pad=58)
        ax[8].set_xlabel('seconds')
        for a in ax[5:8]:
            a.tick_params(labelbottom=False)
    m, r = events[events.source == 'mocap'], events[events.source == 'radar']
    fig.suptitle(f'{session.person} / {session.scenario}: postural transitions, radar and MoCap', x=0.06, y=0.995,
                 ha='left', fontsize=12, fontweight='bold')
    fig.text(0.06, 0.965, f'Transitions in the evaluated time: MoCap {len(m)}, radar {len(r)}, matched (same kind, '
             f'within 1 s) {int(m.matched.sum())}.', fontsize=9, color=style.INK_2)
    fig.legend(handles=LEGEND, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.09 if span else 0.2)
    fig.savefig(path, dpi=110)
    plt.close(fig)
