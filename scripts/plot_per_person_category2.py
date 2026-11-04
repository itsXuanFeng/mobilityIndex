"""
Category 2 per person: signed error in % of the MoCap value, 100 x (radar - MoCap) / MoCap per recording, of the
number of bouts, the mean bout duration, the SD of the bout duration and the fragmentation index. Bars: the person's
mean over their recordings (red: radar too high, blue: too low); dots: one recording each. All on one side = a
systematic error for that person. One y-scale for all panels, so the metrics can be compared.
Reads OUTPUT_ROOT/category2/metrics.csv and bouts.csv: run scripts/evaluate_category2.py first.

    python scripts/plot_per_person_category2.py

Writes OUTPUT_ROOT/plots/per_person/per_person_bouts_bias.png
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
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.plots import style  # noqa: E402

OVER, UNDER = '#e34948', '#2a78d6'           # diverging pair of the reference palette: radar too high / too low
IN = os.path.join(OUTPUT_ROOT, 'category2')
OUT = os.path.join(OUTPUT_ROOT, 'plots', 'per_person')
PANELS = [  # metric, title
    ('n_bouts', 'Number of bouts'),
    ('bout_duration_mean', 'Mean bout duration'),
    ('bout_duration_sd', 'SD of bout duration'),
    ('fragmentation_index', 'Fragmentation index'),
]


def signed_errors():
    """per recording and metric: person, radar - MoCap in % of MoCap. The SD of the bout duration comes from the
    bouts"""
    mt = pd.read_csv(os.path.join(IN, 'metrics.csv'))
    b = pd.read_csv(os.path.join(IN, 'bouts.csv'))
    sd = b.groupby(['rec', 'source']).duration_s.std(ddof=1).unstack('source').reset_index()
    sd = sd.assign(metric='bout_duration_sd')
    mt = pd.concat([mt[['rec', 'metric', 'radar', 'mocap']], sd[['rec', 'metric', 'radar', 'mocap']]])
    return mt.assign(person=mt.rec.str.split('/').str[0], err=100 * (mt.radar - mt.mocap) / mt.mocap)


if __name__ == '__main__':
    e = signed_errors()
    persons = sorted(e.person.unique(), key=natural_key)
    pos = {p: i for i, p in enumerate(persons)}
    x = np.arange(len(persons))
    style.apply()
    lim = np.ceil(1.05 * e[e.metric.isin([k for k, _ in PANELS])].err.abs().max() / 10) * 10   # headroom for the dots
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for ax, (k, title) in zip(axes, PANELS):
        g = e[e.metric == k]
        mean = g.groupby('person').err.mean().reindex(persons).values
        ax.bar(x, mean, 0.6, color=np.where(mean >= 0, OVER, UNDER), edgecolor=style.SURFACE, linewidth=1)
        ax.axhline(0, color=style.INK_2, lw=1)
        jitter = (g.groupby('person').cumcount() - g.groupby('person').rec.transform('size').sub(1) / 2) * 0.08
        ax.plot(g.person.map(pos) + jitter, g.err, 'o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=1)
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:+.0f}%' if v else '0%')
        ax.set_ylabel('radar − MoCap\n(% of MoCap)')
        ax.grid(axis='x', visible=False)
        ax.set_title(f'{title}: mean signed error {g.err.mean():+.1f}%, mean absolute error {g.err.abs().mean():.1f}% '
                     f'(per recording)', loc='left')
    axes[-1].set_xticks(x, [p.replace('person_', 'p') for p in persons])
    axes[-1].set_xlim(-0.7, len(persons) - 0.3)
    axes[0].plot([], [], 'o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=1, label='one recording')
    axes[0].bar([np.nan], [np.nan], color=OVER, label='person mean: radar too high')
    axes[0].bar([np.nan], [np.nan], color=UNDER, label='person mean: radar too low')
    axes[0].legend(ncol=3, loc='lower right', bbox_to_anchor=(1, 1.0), borderaxespad=0.2)
    fig.suptitle('Category 2: signed error per person (148 recordings, 27 people)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'per_person_bouts_bias.png')
    fig.savefig(path, dpi=160)
    print(f'wrote {path}')
    t = e.groupby(['metric', 'person']).err.mean().unstack('metric')[[k for k, _ in PANELS]]
    print(t.reindex(persons).round(1).to_string())
