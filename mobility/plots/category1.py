"""
Category 1 result figures (scripts/plot_category1.py), from the tables of the category1 run:
    timeline_figure         one recording: MoCap RMS joint speed and radar activity level per epoch, their thresholds,
                            and where the two active / inactive calls agree
    roc_figure              ROC of the radar activity level against MoCap active: all epochs pooled, one curve per
                            participant, and each participant's calibrated threshold
    agreement_figure        per recording: radar vs MoCap active time, and the radar energy converted to MoCap units
                            vs the MoCap energy, against the identity line
    per_participant_figure  per participant: AUC and F1 per epoch, and the error of active time and of the energy
                            per recording; one dot per participant, the median marked
No participant IDs on the figures: they are for the public docs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from . import style


def _dot(ax, x, y, colour, **kw):
    ax.scatter(x, y, s=18, color=colour, edgecolor=style.SURFACE, linewidth=0.6, zorder=2, **kw)


def timeline_figure(ep, rec, mocap_threshold, path):
    """ep: epochs.csv.gz of the category1 run (evaluation window); rec: the recording to show"""
    style.apply()
    g = ep[ep.rec == rec].sort_values('t')
    t = (g.t - g.t.iloc[0]).values / 60
    gap = np.r_[False, np.diff(g.t.values) > 1.5]          # epochs outside the evaluation window: break the lines

    def broken(x):
        x = np.asarray(x, float).copy()
        x[gap] = np.nan
        return x

    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[2, 2, 0.8]))
    axes[0].plot(t, broken(g.speed), color=style.MOCAP, lw=1)
    axes[0].axhline(mocap_threshold, color=style.INK_2, lw=0.8, ls='--')
    axes[0].set_ylabel('MoCap: RMS joint\nspeed (m/s)')
    axes[1].plot(t, broken(g.n_cells), color=style.RADAR, lw=1)
    axes[1].axhline(g.radar_threshold.iloc[0], color=style.INK_2, lw=0.8, ls='--')
    axes[1].set_ylabel('radar: activity\nlevel (log count)')
    ra, ma = (g.n_cells > g.radar_threshold).values, g.active.values
    state = np.where(ra & ma, 0, np.where(~ra & ~ma, 1, np.where(ma, 2, 3)))
    colours = [style.AGREE_ACTIVE, style.AGREE_INACTIVE, style.ONLY_MOCAP, style.ONLY_RADAR]
    names = ['both active', 'both inactive', 'only MoCap active', 'only radar active']
    width = np.median(np.diff(t))
    for k in range(4):
        axes[2].bar(t[state == k], 1, width=width, color=colours[k], label=names[k], align='edge', linewidth=0)
    axes[2].set_yticks([])
    axes[2].grid(False)
    axes[2].set_ylabel('calls')
    axes[2].set_xlabel('time (min)')
    axes[0].set_title('MoCap and radar per epoch, one recording (dashed: thresholds)')
    fig.legend(*axes[2].get_legend_handles_labels(), ncol=4, loc='outside lower center')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    agree = (ra == ma).mean()
    return dict(agreement=agree)


def _operating_point(g):
    pred = g.n_cells > g.radar_threshold
    return (pred & ~g.active).sum() / (~g.active).sum(), (pred & g.active).sum() / g.active.sum()


def roc_figure(ep, path):
    """ep: epochs.csv.gz of the category1 run"""
    style.apply()
    ep = ep[ep.n_cells.notna()]
    curves = [(roc_curve(g.active, g.n_cells), roc_auc_score(g.active, g.n_cells)) for _, g in ep.groupby('person')]
    (fpr, tpr, _), auc = roc_curve(ep.active, ep.n_cells), roc_auc_score(ep.active, ep.n_cells)
    pts = np.array([_operating_point(g) for _, g in ep.groupby('person')])
    aucs = [a for _, a in curves]
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    ins = ax.inset_axes([0.42, 0.3, 0.52, 0.42])
    for a in (ax, ins):
        for (f, tp, _), _ in curves:
            a.plot(f, tp, color=style.BASELINE, lw=0.8, zorder=1)
        a.plot(fpr, tpr, color=style.RADAR, lw=2, zorder=3)
        a.plot(pts[:, 0], pts[:, 1], 'o', ms=4 if a is ax else 3.5, mfc=style.SURFACE, mec=style.INK, mew=0.8,
               zorder=4)
    ax.plot([0, 1], [0, 1], color=style.BASELINE, lw=0.8, ls=':')
    ax.plot([], [], color=style.RADAR, lw=2, label=f'all epochs pooled (AUC {auc:.3f})')
    ax.plot([], [], color=style.BASELINE, lw=0.8, label=f'one participant (AUC {min(aucs):.2f}–{max(aucs):.2f})')
    ax.plot([], [], 'o', ms=4, mfc=style.SURFACE, mec=style.INK, mew=0.8, label='calibrated threshold, one participant')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ins.set_xlim(0, 0.25)
    ins.set_ylim(0.75, 1.0)
    ins.tick_params(labelsize=7)
    ax.indicate_inset_zoom(ins, edgecolor=style.MUTED)
    ax.set_xlabel('false positive rate (MoCap inactive, radar active)')
    ax.set_ylabel('true positive rate (MoCap active, radar active)')
    ax.set_title('Active / inactive per epoch: ROC')
    ax.legend(loc='lower right')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(auc=auc, auc_min=min(aucs), auc_max=max(aucs), fpr_median=np.median(pts[:, 0]),
                tpr_median=np.median(pts[:, 1]))


def agreement_figure(times, energy, path):
    """times: active_time rows of metrics.csv; energy: mobility.evaluation.energy_conversion of the energy rows, with
    the raw radar energy added as column radar"""
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    d = times.radar - times.mocap
    e = energy.converted - energy.mocap
    rho = spearmanr(energy.radar, energy.mocap)[0]     # on the raw radar energy: the conversion factor varies by fold
    err = 100 * e.abs().mean() / energy.mocap.mean()
    panels = [(axes[0], times.mocap, times.radar, 'Active time per recording', 'MoCap (min)', 'radar (min)',
               f'mean absolute error {d.abs().mean():.2f} min\nbias {d.mean():+.2f} min'),
              (axes[1], energy.mocap, energy.converted, 'Total activity energy per recording',
               'MoCap (m²/s² · s)', 'radar, converted to MoCap units',
               f'Spearman ρ {rho:.2f}\nerror {err:.1f}% of the mean')]
    for ax, x, y, title, xl, yl, txt in panels:
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=style.INK_2, lw=1, zorder=1)
        _dot(ax, x, y, style.RADAR)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=8, color=style.INK_2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(time_mae=d.abs().mean(), time_bias=d.mean(), spearman=rho, energy_error_pct=err)


def per_participant_table(ep, times, energy):
    """per participant: AUC of the radar level and F1 of the active calls over their epochs (ep: epochs.csv.gz), and
    the mean absolute error of active time and of the converted energy over their recordings, in % of their mean
    MoCap value"""
    rows = []
    ep = ep[ep.n_cells.notna()]
    for p, g in ep.groupby('person'):
        t, e = times[times.person == p], energy[energy.person == p]
        rows.append(dict(person=p, auc=roc_auc_score(g.active, g.n_cells),
                         f1=f1_score(g.active, g.n_cells > g.radar_threshold),
                         time_error_pct=100 * (t.radar - t.mocap).abs().mean() / t.mocap.mean(),
                         energy_error_pct=100 * (e.converted - e.mocap).abs().mean() / e.mocap.mean()))
    return pd.DataFrame(rows)


PARTICIPANT_PANELS = [  # (title, x label, x range, [(column, row label)])
    ('Per epoch: active / inactive', 'score', (0.8, 1.0), [('auc', 'AUC'), ('f1', 'F1')]),
    ('Per recording: error', "mean absolute error, % of the participant's MoCap value", (0, None),
     [('time_error_pct', 'active time'), ('energy_error_pct', 'activity energy')]),
]


def per_participant_figure(table, path):
    """table: per_participant_table. One dot per participant (unlabelled), jittered vertically; the median as a
    vertical bar with its value"""
    style.apply()
    fig, axes = plt.subplots(1, len(PARTICIPANT_PANELS), figsize=(10, 3.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    out = {}
    for ax, (title, xlabel, xlim, rows) in zip(axes, PARTICIPANT_PANELS):
        for i, (col, label) in enumerate(rows):
            x = table[col].values
            y = i + rng.uniform(-0.15, 0.15, len(x))
            ax.plot(x, y, 'o', ms=6, color=style.RADAR, mec=style.SURFACE, mew=0.8, alpha=0.9, zorder=2)
            med = np.median(x)
            ax.plot([med, med], [i - 0.3, i + 0.3], color=style.INK, lw=2, zorder=3)
            mean = np.mean(x)
            ax.text(med, i + 0.33, f'median {med:.2f}, mean {mean:.2f}' if xlim[1] == 1.0
                    else f'median {med:.1f}%, mean {mean:.1f}%', ha='center',
                    va='top', fontsize=8, color=style.INK_2)
            out[col] = dict(median=med, mean=mean, min=x.min(), max=x.max())
        ax.set_yticks(range(len(rows)), [label for _, label in rows])
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.set_xlim(*xlim)
        if xlim[1] is None:
            ax.set_xlim(0, max(table[c].max() for c, _ in rows) * 1.08)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    fig.suptitle(f'Per participant ({len(table)} participants, one dot each; bar: median)', x=0.01, ha='left',
                 fontsize=10, color=style.INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out
