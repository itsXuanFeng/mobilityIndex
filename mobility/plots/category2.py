"""
Category 2 result figures (scripts/plot_category2.py), from the tables of the category2 run:
    formation_figure        one segment of one recording: the radar's distance-corrected level with the entry, exit and
                            confirmation thresholds, the candidate runs kept as bouts or rejected, the radar's posture
                            transitions, and the radar and MoCap bouts
    matching_figure         matched bouts: share whose start and end lie within k s of MoCap; per recording bout
                            precision, recall, F1
    metrics_figure          per recording: the four metrics, radar vs MoCap, against the identity line
    per_participant_figure  per participant: bout F1, and the error of each metric over their recordings
    checks_figure           bouts against the script labels (F1, radar vs MoCap); bout count error in and out of bed
No participant IDs on the figures: they are for the public docs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..utils import runs
from . import style

METRICS = [  # metric, title, unit
    ('n_bouts', 'Number of bouts', ''),
    ('bout_duration_mean', 'Mean bout duration', 's'),
    ('bout_duration_var', 'Variance of bout duration', 's²'),
    ('fragmentation_index', 'Fragmentation index', ''),
]


def _strip(ax, i, x, rng, fmt):
    """one row of dots (one per participant) at position i, with the median as a bar; median and mean written below"""
    y = i + rng.uniform(-0.15, 0.15, len(x))
    ax.plot(x, y, 'o', ms=6, color=style.RADAR, mec=style.SURFACE, mew=0.8, zorder=2)
    med, mean = np.median(x), np.mean(x)
    ax.plot([med, med], [i - 0.3, i + 0.3], color=style.INK, lw=2, zorder=3)
    ax.text(med, i + 0.33, f'median {fmt.format(med)}, mean {fmt.format(mean)}', ha='center', va='top', fontsize=8,
            color=style.INK_2)
    return dict(median=med, mean=mean)


def formation_figure(level, valid, candidates, radar_bout, mocap_bout, transitions_s, thresholds, path, length=100,
                     start=None):
    """level, valid, candidates (runs after the hysteresis, before the confirmation), radar_bout, mocap_bout: per
    epoch of one recording (1 s); transitions_s: radar posture transition times (s since the first epoch);
    thresholds: (exit, entry, confirmation). The segment starts at `start` (epoch), or the first `length` s with most
    evaluated epochs, a rejected candidate and a bout confirmed only by a transition."""
    exit_, entry, conf = thresholds
    if start is None:
        rejected = [s for s, e in runs(candidates) if not radar_bout[s:e].any()]
        by_transition = [s for s, e in runs(radar_bout) if level[s:e].max() <= conf
                         and any(s - 1 <= t < e + 1 for t in transitions_s)]
        best = -1
        for s0 in range(0, len(level) - length, 5):
            if valid[s0:s0 + length].mean() < 0.9:
                continue
            score = sum(s0 <= s < s0 + length for s in rejected) + 2 * sum(s0 <= s < s0 + length for s in by_transition)
            if score > best:
                best, start = score, s0
    seg = slice(start, start + length)
    x = np.arange(length + 1)
    style.apply()
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[3, 1.1]))
    ax = axes[0]
    lv = np.asarray(level[seg], float).copy()
    lv[~valid[seg]] = np.nan
    for s, e in runs(candidates[seg]):
        kept = radar_bout[seg][s:e].any()
        ax.axvspan(s, e, color=style.RADAR if kept else style.MUTED, alpha=0.15 if kept else 0.3, lw=0)
    ax.step(x, np.r_[lv, lv[-1]], where='post', color=style.RADAR, lw=1.3)
    for y, lab, ls in [(conf, 'confirmation  θ + κ', ':'), (entry, 'entry  θ', '--'), (exit_, 'exit  θ − κ', ':')]:
        ax.hlines(y, 0, length, color=style.INK_2, lw=0.9, ls=ls)
        ax.text(length + 1.5, y, lab, va='center', fontsize=8, color=style.INK_2)
    top = np.nanmax(lv) + 0.35
    for t in transitions_s:
        if start <= t < start + length:
            ax.plot(t - start, top, 'v', color=style.INK, ms=6)
    ax.set_ylim(None, top + 0.25)
    ax.set_ylabel('radar activity level\n(distance-corrected)')
    ax.set_title('Bout formation in one recording')
    fig.legend(handles=[Patch(color=style.RADAR, alpha=0.3, label='candidate kept as a bout'),
                        Patch(color=style.MUTED, alpha=0.45, label='candidate rejected (not confirmed)'),
                        plt.Line2D([], [], ls='none', marker='v', color=style.INK, label='radar posture transition')],
               ncol=3, loc='outside lower center')
    ax = axes[1]
    for row, b, colour in [(1, radar_bout[seg], style.RADAR), (0, mocap_bout[seg], style.MOCAP)]:
        for s, e in runs(b):
            ax.barh(row, e - s, left=s, height=0.6, color=colour)
    ax.set_yticks([0, 1], ['MoCap bouts', 'radar bouts'])
    ax.grid(False)
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, length + 15)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return start


def matching_figure(bouts, scores, path):
    """bouts: bouts.csv; scores: bout_scores.csv of the category2 run"""
    m = bouts[(bouts.source == 'radar') & bouts.onset_err_s.notna()]
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    out = {}
    ax = axes[0]
    kmax = int(max(m.onset_err_s.abs().max(), m.offset_err_s.abs().max()))
    k = np.arange(0, kmax + 2)
    lines = []
    # labelled points (epoch k, x offset, y offset, alignment), placed clear of both curves and the axis
    marks = dict(onset_err_s=[(0, 0.15, 0.012, 'left'), (1, -0.15, 0.0, 'right')],
                 offset_err_s=[(0, 0.15, -0.014, 'left'), (1, -0.15, 0.0, 'right'), (2, 0.15, -0.012, 'left'),
                               (3, 0.15, -0.012, 'left')])
    for col, name, colour in [('onset_err_s', 'start', style.SLOTS[0]), ('offset_err_s', 'end', style.SLOTS[1])]:
        a = m[col].abs()
        share = np.array([(a <= i).mean() for i in k])        # the full cumulative distribution, up to 100%
        full = int(a.max())
        ax.step(k, share, where='post', color=colour, lw=2, label=f'{name} of the bout')
        for i, dx, dy, ha in marks[col]:
            ax.plot(i, share[i], 'o', color=colour, ms=6)
            ax.text(i + dx, share[i] + dy, f'{share[i]:.0%}', ha=ha, va='center', fontsize=8, color=style.INK)
        ax.plot(full, 1.0, 'o', color=colour, ms=6)
        lines.append(f'every {name} within {full} s')
        out[col] = dict(exact=round(share[0], 3), within_1s=round(share[1], 3), within_3s=round(share[3], 3),
                        all_within_s=full)
    ax.text(0.97, 0.3, '\n'.join(lines), transform=ax.transAxes, ha='right', va='center', fontsize=8,
            color=style.INK_2)
    ax.set_xlim(-0.5, kmax + 1.5)
    ax.set_xticks(range(0, kmax + 2, 2))
    ax.set_ylim(0.7, 1.03)
    ax.yaxis.set_major_formatter(lambda y, _: f'{y:.0%}')
    ax.set_xlabel('absolute difference, radar − MoCap (s; 0 = same epoch)')
    ax.set_ylabel('share of matched bouts within')
    ax.legend(loc='lower right')
    ax.set_title(f'Bout boundaries ({len(m):,} matched bouts)')
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, (col, lab) in enumerate([('bout_precision', 'precision'), ('bout_recall', 'recall'), ('bout_f1', 'F1')]):
        v = scores[col].dropna()
        ax.plot(rng.uniform(-0.18, 0.18, len(v)) + i, v, 'o', ms=3, color=style.RADAR, alpha=0.6, mec='none')
        ax.plot([i - 0.3, i + 0.3], [v.median()] * 2, color=style.INK, lw=2)
        ax.text(i, v.median() + 0.008, f'{v.median():.2f}', ha='center', va='bottom', fontsize=8, color=style.INK)
        out[col] = v.median()
    ax.set_xticks(range(3), ['precision', 'recall', 'F1'])
    ax.set_ylim(0.6, 1.02)
    ax.grid(axis='x', visible=False)
    ax.set_title('Bout matching, per recording')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    out['matched_bouts'] = len(m)
    return out


FORMATS = dict(n_bouts='.2f', bout_duration_mean='.2f', bout_duration_var='.1f', fragmentation_index='.3f')


def metrics_figure(mt, path):
    """mt: metrics.csv of the category2 run. Per metric: mean absolute error and bias over the recordings, in the
    metric's unit; the error also in % of the mean MoCap value"""
    style.apply()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    out = {}
    for ax, (k, name, unit) in zip(axes, METRICS):
        g = mt[mt.metric == k]
        jit = (lambda n: rng.uniform(-0.25, 0.25, n)) if k == 'n_bouts' else (lambda n: 0)
        lo, hi = min(g.radar.min(), g.mocap.min()) * 0.95, max(g.radar.max(), g.mocap.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=style.INK_2, lw=1)
        ax.scatter(g.mocap + jit(len(g)), g.radar + jit(len(g)), s=14, color=style.RADAR, edgecolor=style.SURFACE,
                   linewidth=0.5, zorder=2)
        d = g.radar - g.mocap
        mae, bias, pct = d.abs().mean(), d.mean(), 100 * d.abs().mean() / g.mocap.mean()
        out[k] = (mae, bias, pct)
        fmt, u_ = FORMATS[k], f' {unit}' if unit else ''
        ax.text(0.04, 0.96, f'mean absolute error {mae:{fmt}}{u_} ({pct:.1f}%)\nbias {bias:+{fmt}}{u_}',
                transform=ax.transAxes, va='top', fontsize=8, color=style.INK_2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        u = f' ({unit})' if unit else ''
        ax.set_xlabel('MoCap' + u)
        ax.set_ylabel('radar' + u)
        ax.set_title(name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


def per_participant_table(mt, scores, person_of):
    """per participant: bout F1 pooled over their recordings, and the mean absolute error of each metric over their
    recordings in % of their mean MoCap value"""
    sc = scores.assign(person=person_of(scores.rec))
    mt = mt.assign(person=person_of(mt.rec))
    rows = []
    for p, g in sc.groupby('person'):
        row = dict(person=p, bout_f1=2 * g.matched.sum() / (g.mocap_bouts.sum() + g.radar_bouts.sum()))
        for k, _, _ in METRICS:
            h = mt[(mt.person == p) & (mt.metric == k)]
            row[k] = 100 * (h.radar - h.mocap).abs().mean() / h.mocap.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def per_participant_figure(table, path):
    """table: per_participant_table. One dot per participant (unlabelled); the median as a bar"""
    style.apply()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 2.3]))
    out = {}
    ax = axes[0]
    out['bout_f1'] = _strip(ax, 0, table.bout_f1.values, rng, '{:.2f}')
    ax.set_yticks([0], ['bout F1'])
    ax.set_ylim(-0.6, 0.7)
    ax.invert_yaxis()
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel('score')
    ax.grid(axis='y', visible=False)
    ax.set_title('Bout matching')
    ax = axes[1]
    for i, (k, _, _) in enumerate(METRICS):
        out[k] = _strip(ax, i, table[k].values, rng, '{:.1f}%')
    ax.set_yticks(range(len(METRICS)), [n.lower() for _, n, _ in METRICS])
    ax.set_ylim(-0.6, len(METRICS) - 0.25)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel("mean absolute error, % of the participant's MoCap value")
    ax.grid(axis='y', visible=False)
    ax.set_title('Metric error')
    fig.suptitle(f'Per participant ({len(table)} participants, one dot each; bar: median)', x=0.01, ha='left',
                 fontsize=10, color=style.INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


def checks_figure(epoch_scores, bouts, path):
    """epoch_scores: epoch_scores.csv (F1 of the in-bout calls against the labels, radar and MoCap); bouts: bouts.csv
    (in_bed per bout)"""
    style.apply()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.3]))
    out = {}
    ax = axes[0]
    e = epoch_scores.dropna(subset=['f1_labels_radar', 'f1_labels_mocap'])
    lo = min(e.f1_labels_radar.min(), e.f1_labels_mocap.min()) - 0.02
    ax.plot([lo, 1], [lo, 1], color=style.INK_2, lw=1)
    ax.scatter(e.f1_labels_mocap, e.f1_labels_radar, s=14, color=style.RADAR, edgecolor=style.SURFACE, linewidth=0.5,
               zorder=2)
    out['f1_labels'] = (e.f1_labels_mocap.median(), e.f1_labels_radar.median())
    ax.text(0.04, 0.96, f'median, MoCap: {e.f1_labels_mocap.median():.3f}\nmedian, radar: {e.f1_labels_radar.median():.3f}',
            transform=ax.transAxes, va='top', fontsize=8, color=style.INK_2)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('MoCap bouts: F1 against the labels')
    ax.set_ylabel('radar bouts: F1 against the labels')
    ax.set_title('Bouts against the script labels')
    ax = axes[1]
    cnt = bouts.groupby(['rec', 'in_bed', 'source']).size().unstack('source', fill_value=0).reset_index()
    cnt['diff'] = cnt.radar - cnt.mocap
    for i, (flag, name) in enumerate([(False, 'out of bed'), (True, 'in bed')]):
        v = cnt[cnt.in_bed == flag]['diff'].values
        ax.plot(v + rng.uniform(-0.25, 0.25, len(v)), i + rng.uniform(-0.15, 0.15, len(v)), 'o', ms=4,
                color=style.RADAR, alpha=0.6, mec='none')
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=2)
        ax.text(v.mean(), i + 0.33, f'mean {v.mean():+.2f}', ha='center', va='top', fontsize=8, color=style.INK_2)
        out[name] = v.mean()
    ax.axvline(0, color=style.BASELINE, lw=1)
    ax.set_yticks([0, 1], ['out of bed', 'in bed'])
    ax.set_ylim(1.6, -0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('number of bouts per recording, radar − MoCap (one dot per recording)')
    ax.set_title('Bout count in bed and out of bed')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out
