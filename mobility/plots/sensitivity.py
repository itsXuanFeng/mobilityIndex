"""Category 1 active / inactive (n_cells) against its two hand-set parameters (scripts/sensitivity_floor.py)."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from ..config import ABOVE_FLOOR_DB, FLOOR_PERCENTILE
from . import style

SCALE = (0.80, 1.00)     # one colour scale for every panel, so small differences look small


def _heatmap(ax, grid, title, fmt='{:.3f}'):
    cmap = LinearSegmentedColormap.from_list('blues', style.BLUES)
    lo, hi = SCALE
    ax.imshow(grid.values, cmap=cmap, vmin=lo, vmax=hi, aspect='auto', origin='lower')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, p in enumerate(grid.index):
        for j, m in enumerate(grid.columns):
            v = grid.values[i, j]
            shade = cmap(np.clip((v - lo) / (hi - lo), 0, 1))
            dark = 0.2126 * shade[0] + 0.7152 * shade[1] + 0.0722 * shade[2] < 0.45
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color=style.SURFACE if dark else style.INK)
    i, j = list(grid.index).index(FLOOR_PERCENTILE), list(grid.columns).index(ABOVE_FLOOR_DB)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=style.INK, lw=1.8))
    ax.set_xticks(range(len(grid.columns)), [f'{m:g}' for m in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f'{p}th' for p in grid.index])
    ax.set_xlabel('margin above the floor (dB)')
    ax.set_title(title)


def floor_sensitivity_figure(res, path):
    """AUC and F1 heatmaps over (noise-floor percentile, margin): res is floor_sensitivity.csv"""
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    panels = [('auc_median', 'AUC, median over recordings'), ('f1_median', 'F1, median over recordings'),
              ('f1_p10', 'F1, 10th percentile (worst recordings)')]
    for ax, (k, title) in zip(axes, panels):
        _heatmap(ax, res.pivot(index='percentile', columns='margin_db', values=k), title)
    axes[0].set_ylabel('noise-floor percentile')
    fig.suptitle(f'Active / inactive (n_cells) vs its two parameters, {res.recordings.iloc[0]} recordings, '
                 f'leave-person-out thresholds.\nOutlined: current setting '
                 f'({FLOOR_PERCENTILE}th percentile, {ABOVE_FLOOR_DB:g} dB). Colour: one scale for all panels, '
                 f'{SCALE[0]:.2f} (light) to {SCALE[1]:.2f} (dark).',
                 x=0.01, ha='left', fontsize=9, color=style.INK_2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
