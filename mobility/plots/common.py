"""Pieces shared by the figures over time: the label row, strip rows, hatching of what isn't evaluated, episodes."""
import matplotlib.pyplot as plt
import numpy as np

from ..labels import ACTIVE, INACTIVE, label_class
from ..utils import runs
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


def strip_axes(ax, label):
    """a row without a y scale, its label to the left"""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=8)


def label_spans(ax, labels, t0, scale, names_until=None):
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


def hatch_unevaluated(ax, x, width, evaluated):
    """hatch the runs of samples (at x, each `width` wide) that are not evaluated"""
    for i0, i1 in runs(~np.asarray(evaluated, bool)):
        ax.axvspan(x[i0], x[i1 - 1] + width, facecolor='none', edgecolor=style.HATCH, hatch='////', linewidth=0,
                   zorder=0)


def episode_span(labels, first, last, pad=3.0):
    """(start, end) epoch s from the first label `first` to the next label `last`, padded; None if absent"""
    lab = labels.label.str.lower().values
    i = np.flatnonzero(lab == first)
    if not len(i):
        return None
    j = [k for k in np.flatnonzero(lab == last) if k >= i[0]]
    if not j:
        return None
    return labels.start.values[i[0]] - pad, labels.end.values[j[0]] + pad
