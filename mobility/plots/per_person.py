"""
Per-person figures from the evaluation tables:
    scores_figure, energy_error_figure, energy_bias_figure   Category 1 (scripts/plot_per_person.py)
    bouts_bias_figure                                         Category 2 (scripts/plot_per_person_category2.py)
Signed errors: bars are the person's mean (red: radar too high, blue: too low), dots one recording each. All on one
side = a systematic error for that person.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import style

BOUT_PANELS = [  # metric, title
    ('n_bouts', 'Number of bouts'),
    ('bout_duration_mean', 'Mean bout duration'),
    ('bout_duration_sd', 'SD of bout duration'),
    ('fragmentation_index', 'Fragmentation index'),
]


def _short(persons):
    return [p.replace('person_', 'p') for p in persons]


def _person_axis(ax, labels):
    x = np.arange(len(labels))
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.grid(axis='x', visible=False)
    return x


def _signed_bars(ax, x, mean, recs, pos, values, **dots):
    """bars at x: each person's mean signed error; dots: the recordings (rows of recs with person and rec), jittered
    around their person's position pos[person]"""
    ax.bar(x, mean, 0.6, color=np.where(mean >= 0, style.OVER, style.UNDER), edgecolor=style.SURFACE, linewidth=1)
    ax.axhline(0, color=style.INK_2, lw=1)
    jitter = (recs.groupby('person').cumcount() - recs.groupby('person').rec.transform('size').sub(1) / 2) * 0.08
    ax.plot(recs.person.map(pos) + jitter, recs[values], 'o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=1, **dots)


def scores_figure(pp, path):
    """AUC and F1 of the active / inactive calls per person (per_person.csv of Category 1)"""
    style.apply()
    fig, ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
    x = _person_axis(ax, _short(pp.person))
    w = 0.27
    for i, (k, name) in enumerate([('auc', 'AUC (active & inactive)'), ('f1_active', 'F1 active'),
                                   ('f1_inactive', 'F1 inactive')]):
        ax.bar(x + (i - 1) * w, pp[k], w, color=style.SLOTS[i], edgecolor=style.SURFACE, linewidth=1, label=name)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('score')
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title('Active / inactive: AUC and F1 per person', pad=22)
    fig.savefig(path, dpi=160)


def energy_error_figure(pp, baseline, path):
    """total activity energy error per person (per_person.csv), next to the no-radar baseline's error (%)"""
    style.apply()
    fig, ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
    x = _person_axis(ax, _short(pp.person))
    ax.bar(x, pp.energy_error_pct, 0.6, color=style.SLOTS[0], edgecolor=style.SURFACE, linewidth=1)
    ax.axhline(baseline, color=style.INK_2, lw=1, label=f'no-radar baseline (all recordings): {baseline:.1f}%')
    ax.legend(loc='upper left')
    for i in pp.energy_error_pct.nlargest(2).index:                 # label the two extremes only
        ax.text(x[i], pp.energy_error_pct[i] + 0.6, f'{pp.energy_error_pct[i]:.1f}', ha='center', va='bottom',
                fontsize=8, color=style.INK)
    ax.set_ylim(0, max(pp.energy_error_pct.max(), baseline) * 1.15)
    ax.set_ylabel('error, % of MoCap energy')
    ax.set_title('Total activity energy: error per person')
    fig.savefig(path, dpi=160)


def energy_bias_figure(pp, conversion, path):
    """signed total activity energy error per person (per_person.csv) and per recording (conversion: rows of
    mobility.evaluation.energy_conversion with bias_pct)"""
    style.apply()
    pos = dict(zip(pp.person, range(len(pp))))
    fig, ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
    x = _person_axis(ax, _short(pp.person))
    _signed_bars(ax, x, pp.energy_bias_pct.values, conversion, pos, 'bias_pct', label='one recording')
    ax.bar([np.nan], [np.nan], color=style.OVER, label='person: radar too high')
    ax.bar([np.nan], [np.nan], color=style.UNDER, label='person: radar too low')
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    lim = np.ceil(np.abs(conversion.bias_pct).max() / 10) * 10
    ax.set_ylim(-lim, lim)
    ax.set_ylabel('radar − MoCap, % of MoCap energy')
    ax.set_title('Total activity energy: signed error per person', pad=22)
    fig.savefig(path, dpi=160)


def bouts_bias_figure(errors, persons, path, panels=BOUT_PANELS):
    """Category 2: one panel per metric, signed error per person in % of MoCap. errors: per recording and metric,
    person and err = 100 x (radar - MoCap) / MoCap; persons in plotting order. One y-scale for all panels, so the
    metrics can be compared."""
    e = errors
    pos = {p: i for i, p in enumerate(persons)}
    x = np.arange(len(persons))
    style.apply()
    lim = np.ceil(1.05 * e[e.metric.isin([k for k, _ in panels])].err.abs().max() / 10) * 10   # headroom for the dots
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for ax, (k, title) in zip(axes, panels):
        g = e[e.metric == k]
        _signed_bars(ax, x, g.groupby('person').err.mean().reindex(persons).values, g, pos, 'err')
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:+.0f}%' if v else '0%')
        ax.set_ylabel('radar − MoCap\n(% of MoCap)')
        ax.grid(axis='x', visible=False)
        ax.set_title(f'{title}: mean signed error {g.err.mean():+.1f}%, mean absolute error {g.err.abs().mean():.1f}% '
                     f'(per recording)', loc='left')
    axes[-1].set_xticks(x, _short(persons))
    axes[-1].set_xlim(-0.7, len(persons) - 0.3)
    axes[0].plot([], [], 'o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=1, label='one recording')
    axes[0].bar([np.nan], [np.nan], color=style.OVER, label='person mean: radar too high')
    axes[0].bar([np.nan], [np.nan], color=style.UNDER, label='person mean: radar too low')
    axes[0].legend(ncol=3, loc='lower right', bbox_to_anchor=(1, 1.0), borderaxespad=0.2)
    fig.suptitle('Category 2: signed error per person (148 recordings, 27 people)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    fig.savefig(path, dpi=160)
