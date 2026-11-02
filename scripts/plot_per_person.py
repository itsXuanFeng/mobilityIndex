"""
Category 1 per person, as bar plots: AUC and F1 of the active / inactive calls, and the error of the total activity
energy. Reads OUTPUT_ROOT/category1/per_person.csv and agreement.csv: run scripts/evaluate_category1.py first.

    python scripts/plot_per_person.py

Writes OUTPUT_ROOT/plots/per_person/per_person_scores.png and per_person_energy_error.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import OUTPUT_ROOT  # noqa: E402
from mobility.plots import style  # noqa: E402

SLOTS = ['#2a78d6', '#eb6834', '#1baf7a']     # categorical slots 1-3 of the reference palette: one per metric
IN = os.path.join(OUTPUT_ROOT, 'category1')
OUT = os.path.join(OUTPUT_ROOT, 'plots', 'per_person')


def axis(ax, labels):
    x = np.arange(len(labels))
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.grid(axis='x', visible=False)
    return x


if __name__ == '__main__':
    pp = pd.read_csv(os.path.join(IN, 'per_person.csv'))
    ag = pd.read_csv(os.path.join(IN, 'agreement.csv')).set_index('metric')
    baseline = ag.loc['total_activity_energy', 'baseline_error_pct']
    labels = [p.replace('person_', 'p') for p in pp.person]
    style.apply()
    os.makedirs(OUT, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
    x = axis(ax, labels)
    w = 0.27
    for i, (k, name) in enumerate([('auc', 'AUC (active & inactive)'), ('f1_active', 'F1 active'),
                                   ('f1_inactive', 'F1 inactive')]):
        ax.bar(x + (i - 1) * w, pp[k], w, color=SLOTS[i], edgecolor=style.SURFACE, linewidth=1, label=name)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('score')
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title('Active / inactive: AUC and F1 per person', pad=22)
    fig.savefig(os.path.join(OUT, 'per_person_scores.png'), dpi=160)

    fig, ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
    x = axis(ax, labels)
    ax.bar(x, pp.energy_error_pct, 0.6, color=SLOTS[0], edgecolor=style.SURFACE, linewidth=1)
    ax.axhline(baseline, color=style.INK_2, lw=1, label=f'no-radar baseline (all recordings): {baseline:.1f}%')
    ax.legend(loc='upper left')
    for i in pp.energy_error_pct.nlargest(2).index:                 # label the two extremes only
        ax.text(x[i], pp.energy_error_pct[i] + 0.6, f'{pp.energy_error_pct[i]:.1f}', ha='center', va='bottom',
                fontsize=8, color=style.INK)
    ax.set_ylim(0, max(pp.energy_error_pct.max(), baseline) * 1.15)
    ax.set_ylabel('error, % of MoCap energy')
    ax.set_title('Total activity energy: error per person')
    fig.savefig(os.path.join(OUT, 'per_person_energy_error.png'), dpi=160)
    print(f'wrote {OUT}')
