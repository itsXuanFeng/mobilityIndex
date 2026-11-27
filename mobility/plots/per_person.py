"""
Per-person figures from the evaluation tables:
    scores_figure, energy_error_figure, energy_bias_figure   Category 1 (scripts/plot_per_person.py)
    bouts_bias_figure                                         Category 2 (scripts/plot_per_person_category2.py)
    signed_error_figure with CATEGORY3_PANELS                 Category 3 (scripts/plot_per_person_category3.py)
    setups_bias_figure                                        personal calibration, setups A-C
                                                              (scripts/plot_per_person_personal.py)
Signed errors: bars are the person's mean (red: radar too high, blue: too low), dots one recording each. All on one
side = a systematic error for that person.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import style

BOUT_PANELS = [  # metric, title: the four Category 2 metrics
    ('n_bouts', 'Number of bouts'),
    ('bout_duration_mean', 'Mean bout duration'),
    ('bout_duration_var', 'Variance of bout duration'),
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


CATEGORY3_PANELS = [  # metric, title, unit (Category 3: signed error in the metric's own units)
    ('sit_to_stand_count', 'Sit-to-stand transitions', 'transitions'),
    ('stand_to_sit_count', 'Stand-to-sit transitions', 'transitions'),
    ('lie_to_sit_count', 'Lie-to-sit transitions', 'transitions'),
    ('peak_transition_doppler', 'Peak transition speed', 'm/s'),
    ('standing_time', 'Time standing', 'min'),
    ('sitting_time', 'Time sitting', 'min'),
    ('lying_time', 'Time lying', 'min'),
    ('floor_time', 'Time on the floor', 'min'),
]


def _fmt(v, unit):
    return f'{v:+.1f}%' if unit == '%' else f'{v:+.2f} {unit}'


def signed_error_figure(errors, persons, panels, path, title, unit=None):
    """one panel per metric: signed error radar - MoCap per person (bars: the person's mean; dots: their recordings),
    and an 'all' column on the right: the mean over all recordings, with a whisker from their 10th to 90th
    percentile. errors: per recording and metric, rec, person, err and (for the titles) mocap; persons in plotting
    order; panels: [(metric, title)] with a common `unit` ('%' = in % of MoCap) or [(metric, title, unit)]. Each
    panel has its own y-range, fitted to its recordings (zero included). Recordings with err NaN (in % of MoCap: MoCap
    is 0) are left out, and the panel title says how many."""
    panels = [p if len(p) == 3 else (*p, unit) for p in panels]
    e = errors[errors.metric.isin([k for k, _, _ in panels])]
    pos = {p: i for i, p in enumerate(persons)}
    x = np.arange(len(persons))
    x_all = len(persons) + 0.6
    style.apply()
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.55 * len(panels) + 0.8), sharex=True,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (k, name, u) in zip(axes, panels):
        g = e[e.metric == k]
        n_out = int(g.err.isna().sum())          # no error defined (in % of MoCap: MoCap is 0)
        g = g.dropna(subset=['err'])
        _signed_bars(ax, x, g.groupby('person').err.mean().reindex(persons).values, g, pos, 'err')
        m, (lo, hi) = g.err.mean(), np.percentile(g.err, [10, 90])
        ax.bar([x_all], [m], 0.6, color=style.OVER if m >= 0 else style.UNDER, edgecolor=style.INK, linewidth=1)
        ax.vlines(x_all, lo, hi, color=style.INK, lw=1.2)                  # the mean may lie outside it (counts)
        ax.hlines([lo, hi], x_all - 0.12, x_all + 0.12, color=style.INK, lw=1.2)
        ax.axvline(len(persons) - 0.1, color=style.MUTED, lw=0.6, ls=':')
        ymin, ymax = min(0, g.err.min()), max(0, g.err.max())     # fitted to this panel's recordings, zero included
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 1
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_major_formatter((lambda v, _: f'{v:+.0f}%' if v else '0%') if u == '%' else
                                     (lambda v, _: f'{v:+g}' if v else '0'))
        ax.set_ylabel('radar − MoCap\n' + ('(% of MoCap)' if u == '%' else f'({u})'))
        ax.grid(axis='x', visible=False)
        scale = '' if u == '%' else f'; MoCap median {g.mocap.median():.2f} {u}'
        if n_out:
            scale += f'; {n_out} recording{"s" if n_out > 1 else ""} with MoCap 0 left out'
        ax.set_title(f'{name}: mean signed error {_fmt(m, u)}, mean absolute error {_fmt(g.err.abs().mean(), u)[1:]} '
                     f'(per recording{scale})', loc='left')
    axes[-1].set_xticks(list(x) + [x_all], _short(persons) + ['all'])
    axes[-1].set_xlim(-0.7, x_all + 0.7)
    h = [plt.Line2D([], [], ls='none', marker='o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=1),
         plt.Rectangle((0, 0), 1, 1, color=style.OVER), plt.Rectangle((0, 0), 1, 1, color=style.UNDER),
         plt.Line2D([], [], color=style.INK, lw=1.2)]
    fig.legend(h, ['one recording', 'mean (person, or all recordings): radar too high', 'mean: radar too low',
                   'all: 10th–90th percentile of recordings'], ncol=4, loc='outside lower center')
    fig.suptitle(f'{title}: signed error per person ({e.rec.nunique()} recordings, {len(persons)} people)', x=0.01,
                 ha='left', fontsize=12, fontweight='bold')
    fig.savefig(path, dpi=160)


def bouts_bias_figure(errors, persons, path, panels=BOUT_PANELS):
    """Category 2: one panel per metric, signed error per person in % of MoCap (signed_error_figure). errors: per
    recording and metric, rec, person, mocap and err = 100 x (radar - MoCap) / MoCap; persons in plotting order."""
    signed_error_figure(errors, persons, panels, path, 'Category 2', unit='%')


SETUP_LABELS = {'A': 'A: LOSO', 'B': 'B: LOSO + clip', 'C': 'C: LOSO + other recordings + clip'}


def setups_bias_figure(errors, persons, panels, path, title):
    """personal calibration (scripts/evaluate_personal_calibration.py): one panel per metric, each person's mean signed
    error in % of MoCap under setups A, B and C side by side. errors: per recording, setup and metric, person and
    err = 100 x (radar - MoCap) / MoCap; persons in plotting order; panels: [(metric, title)]. One y-scale for all
    panels."""
    e = errors[errors.metric.isin([k for k, _ in panels])]
    setups = [s for s in SETUP_LABELS if s in set(e.setup)]
    mean = e.groupby(['metric', 'setup', 'person']).err.mean()
    x, w = np.arange(len(persons)), 0.8 / len(setups)
    style.apply()
    lim = np.ceil(1.1 * mean.abs().max() / 5) * 5
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.6 * len(panels) + 0.8), sharex=True,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (k, name) in zip(axes, panels):
        stats = []
        for i, s in enumerate(setups):
            ax.bar(x + (i - (len(setups) - 1) / 2) * w, mean[k, s].reindex(persons).values, w,
                   color=style.SLOTS[i], edgecolor=style.SURFACE, linewidth=0.6, label=SETUP_LABELS[s])
            g = e[(e.metric == k) & (e.setup == s)].err
            stats.append(f'{s} {g.abs().mean():.1f}% ({g.mean():+.1f}%)')
        ax.axhline(0, color=style.INK_2, lw=1)
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:+.0f}%' if v else '0%')
        ax.set_ylabel('radar − MoCap\n(% of MoCap)')
        ax.grid(axis='x', visible=False)
        ax.set_title(f'{name}: mean absolute error (mean signed error) per recording: ' + ', '.join(stats), loc='left')
    axes[-1].set_xticks(x, _short(persons))
    axes[-1].set_xlim(-0.7, len(persons) - 0.3)
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(setups), loc='outside lower center')
    fig.savefig(path, dpi=160)
