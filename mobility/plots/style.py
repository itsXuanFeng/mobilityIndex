"""
Colours and figure defaults. Two series only, MoCap and radar: slots 1-2 of the reference categorical palette
(validated for colour-vision deficiency). Active / inactive are neutral greys, so colour always means "which sensor".
"""
import matplotlib as mpl

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'          # secondary text
MUTED = '#898781'          # axis labels, ticks
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'

MOCAP = '#2a78d6'          # series 1, blue
RADAR = '#eb6834'          # series 2, orange

ACTIVE = '#52514e'         # dark grey: active (both agree, or label class)
INACTIVE = '#e1e0d9'       # light grey: inactive
HATCH = '#c3c2b7'          # outside the evaluation window / label without class

# agreement of the active / inactive calls per epoch: colour = the sensor that alone says "active"
AGREE_ACTIVE, AGREE_INACTIVE = ACTIVE, INACTIVE
ONLY_MOCAP = MOCAP         # radar missed movement MoCap saw
ONLY_RADAR = RADAR         # radar saw movement MoCap didn't


def apply():
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif', 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.edgecolor': BASELINE, 'axes.linewidth': 0.75, 'axes.grid': True, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED,
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'legend.frameon': False, 'legend.fontsize': 8, 'legend.labelcolor': INK_2,
        'lines.linewidth': 1.4, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'hatch.color': HATCH, 'hatch.linewidth': 0.6,
    })
