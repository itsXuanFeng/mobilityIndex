"""
Radar vs MoCap over time: how well the two sensors match, epoch by epoch.

recording_figure   the whole recording: label classes, movement of both sensors, agreement of the active calls,
                   distance to the radar
episodes_figure    close-ups of typical episodes (chair, bed, floor) at frame resolution, with the label names

Movement is shown as the z-score of log energy (radar: distance-corrected moving power p_r4; MoCap: whole-body mean
squared joint speed). The two have different units, so each is standardised over the evaluated epochs of the recording
and they share one axis without a second scale.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..config import COVERAGE_RADIUS, FPS
from ..labels import ACTIVE, INACTIVE, label_class
from ..utils import log_level, nan_smooth
from . import style

EPISODES = [  # title, first label, last label (the first occurrence of each, in order)
    ('At the chair', 'walk towards the chair', 'stand up from the chair'),
    ('In bed', 'sit down on the edge of the bed', 'stand up from the bed'),
    ('Floor and walking', 'act like you pick up a coin from the floor', 'get up and stand upright'),
]

SHORT = {  # label -> short name, for the close-ups
    'walk towards the chair': 'to chair', 'sit down on the chair': 'sit down',
    'start performing eating movements': 'eat', 'stop eating and put down the plate': 'put down',
    'start performing drinking movements': 'drink', 'stop drinking and put down the glass': 'put down',
    'stand up from the chair': 'stand up', 'go to the left side of the bed': 'to bed',
    'go to the right side of the bed': 'to bed', 'sit down on the edge of the bed': 'sit down',
    'lay down on the bed': 'lie down', 'perform anxious frequent movement in the bed': 'restless',
    'stop performing anxious frequent movement and lay on your back': 'settle',
    'turn on you left side and lay still': 'turn', 'turn on you right side and lay still': 'turn',
    'now, turn on your left side and lie still': 'turn', 'now, turn on your right side and lie still': 'turn',
    'lie on your back again and lie still': 'turn back', 'press the button on the wearable': 'button',
    'sit upright in the bed': 'sit up', 'go from upright to lay on the bed': 'lie down',
    'get up and sit on the edge of the bed': 'sit up', 'stand up from the bed': 'stand up',
    'act like you pick up a coin from the floor': 'pick coin', 'wander around in the room': 'wander',
    'lay down on the ground': 'lie on floor', 'get up and stand upright': 'get up',
}

CLASS_COLOUR = {ACTIVE: style.ACTIVE, INACTIVE: style.INACTIVE}

SERIES = [plt.Line2D([], [], color=style.MOCAP, label='MoCap: whole-body movement'),
          plt.Line2D([], [], color=style.RADAR, label='Radar: distance-corrected moving power')]

LEGEND = [  # the "active?" row
    Patch(facecolor=style.AGREE_ACTIVE, label='both active'),
    Patch(facecolor=style.AGREE_INACTIVE, label='both inactive'),
    Patch(facecolor=style.ONLY_MOCAP, label='only MoCap active (radar missed)'),
    Patch(facecolor=style.ONLY_RADAR, label='only radar active'),
    Patch(facecolor=style.SURFACE, edgecolor=style.HATCH, hatch='////', label='not evaluated'),
]


def _runs(mask):
    """(start, end) index pairs of the runs of True"""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2]))


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
    _strip_axes(ax, label)


def _strip_axes(ax, label):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=8)


def _label_spans(ax, labels, t0, scale, names_until=None):
    """label segments coloured by class (dark = active, light = inactive, hatched = not used); with names_until,
    short names of the labels whose middle lies in [0, names_until]"""
    for _, r in labels.iterrows():
        x0, w = (r.start - t0) / scale, (r.end - r.start) / scale
        cls = CLASS_COLOUR.get(label_class(r.label))
        if cls is None:
            ax.add_patch(plt.Rectangle((x0, 0), w, 1, facecolor=style.SURFACE, edgecolor=style.HATCH,
                                       hatch='////', linewidth=0))
        else:
            ax.add_patch(plt.Rectangle((x0, 0), w, 1, facecolor=cls, linewidth=0))
        if names_until is not None and 0 <= x0 + w / 2 <= names_until:
            short = SHORT.get(r.label.lower())
            if short:
                ax.text(x0 + w / 2, 1.08, short, rotation=90, ha='center', va='bottom', fontsize=7.5,
                        color=style.INK_2, clip_on=False)


def _hatch_unevaluated(ax, x, width, evaluated):
    for i0, i1 in _runs(~np.asarray(evaluated, bool)):
        ax.axvspan(x[i0], x[i1 - 1] + width, facecolor='none', edgecolor=style.HATCH, hatch='////', linewidth=0,
                   zorder=0)


def recording_figure(session, pairs, path):
    """the whole recording; pairs from mobility.evaluation.epoch_pairs"""
    style.apply()
    t0 = session.t[0]
    x = (pairs.t.values - t0) / 60                      # minutes since radar start
    w = 1 / 60                                           # one epoch
    ev = pairs.evaluated.values
    fig, ax = plt.subplots(4, 1, figsize=(12, 6.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.32, 2.4, 0.32, 1.0], hspace=0.18))
    _label_spans(ax[0], session.labels, t0, 60)
    _strip_axes(ax[0], 'activity\nlabel')

    zr = _z(pairs.radar_energy.values, ev)
    zm = _z(pairs.mocap_energy.values, ev)
    _hatch_unevaluated(ax[1], x, w, ev)
    ax[1].plot(x, zm, color=style.MOCAP, label='MoCap: whole-body movement')
    ax[1].plot(x, zr, color=style.RADAR, label='Radar: distance-corrected moving power')
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


def _episode_span(labels, first, last, pad=3.0):
    lab = labels.label.str.lower().values
    i = np.flatnonzero(lab == first)
    if not len(i):
        return None
    j = [k for k in np.flatnonzero(lab == last) if k >= i[0]]
    if not j:
        return None
    return labels.start.values[i[0]] - pad, labels.end.values[j[0]] + pad


def episodes_figure(session, pairs, path, episodes=EPISODES):
    """close-ups of typical episodes at frame resolution; the z-scores use the same reference as recording_figure"""
    style.apply()
    spans = [(title, _episode_span(session.labels, a, b)) for title, a, b in episodes]
    spans = [(title, s) for title, s in spans if s is not None]
    win = session.masks['window']
    zr = nan_smooth(_z(session.radar['p_r4'], win), FPS // 2)
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
        _label_spans(ax[0, c], session.labels, a, 1, names_until=b - a)
        ax[0, c].set_xlim(0, b - a)
        _strip_axes(ax[0, c], 'activity\nlabel' if c == 0 else '')
        ax[0, c].set_title(title, pad=58)
        _hatch_unevaluated(ax[1, c], xe, 1, pairs.evaluated.values[e])
        ax[1, c].plot(xs, zm[f], color=style.MOCAP, label='MoCap: whole-body movement')
        ax[1, c].plot(xs, zr[f], color=style.RADAR, label='Radar: distance-corrected moving power')
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
