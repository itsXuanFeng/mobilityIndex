"""
Category 3 result figures (scripts/plot_category3.py), from the tables of the category3 run and the angle check:
    localisation_figure     height error of the radar's nearest moving reflector against the nearest MoCap joint,
                            from the angle of arrival and from the range alone, by distance from the radar axis
    timeline_figure         one recording: posture of both sensors with their transitions; the radar's upper and median
                            heights and the MoCap head height
    posture_figure          posture per epoch: confusion matrix (row %), and F1 per posture and recording
    posture_time_figure     time in each posture per recording, radar vs MoCap
    transitions_figure      transitions: F1 per type, timing error of matched transitions, share found by distance from
                            the radar axis
    peak_figure             peak radial speed: per matched transition, and the metric per recording
    per_participant_figure  per participant: posture and transition F1, posture-time and peak-speed errors
No participant IDs on the figures: they are for the public docs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import style

POSTURE_NAMES = ['standing', 'sitting', 'lying', 'floor']
KINDS = [('sit_to_stand', 'sit → stand'), ('stand_to_sit', 'stand → sit'), ('sit_to_lie', 'sit → lie'),
         ('lie_to_sit', 'lie → sit'), ('to_floor', 'to the floor'), ('from_floor', 'from the floor')]
COUNTED = [('sit_to_stand', 'sit → stand'), ('stand_to_sit', 'stand → sit'), ('lie_to_sit', 'lie → sit')]
OFFAXIS_BINS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]


def _identity_scatter(ax, x, y, text, lo=0, pad=1.05, **kw):
    hi = max(np.nanmax(x), np.nanmax(y)) * pad
    ax.plot([lo, hi], [lo, hi], color=style.INK_2, lw=1, zorder=1)
    ax.scatter(x, y, s=kw.pop('s', 14), color=kw.pop('color', style.RADAR), edgecolor=style.SURFACE, linewidth=0.5,
               zorder=2, **kw)
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va='top', fontsize=8, color=style.INK_2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')


def _strip(ax, i, x, rng, fmt):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    ax.plot(x, i + rng.uniform(-0.15, 0.15, len(x)), 'o', ms=6, color=style.RADAR, mec=style.SURFACE, mew=0.8,
            zorder=2)
    med = np.median(x)
    ax.plot([med, med], [i - 0.3, i + 0.3], color=style.INK, lw=2, zorder=3)
    ax.text(med, i + 0.33, 'median ' + fmt.format(med), ha='center', va='top', fontsize=8, color=style.INK_2)
    return med


def localisation_figure(errors, path, bins=(0, 0.5, 1, 1.5, 2, 3, 5)):
    """errors: errors.csv of the radar_orientation run (per frame: height_angle, height_range, horizontal, offaxis)"""
    style.apply()
    g = errors.groupby(pd.cut(errors.offaxis, list(bins)), observed=True).agg(
        n=('offaxis', 'size'), angle=('height_angle', lambda v: 100 * v.abs().median()),
        rng=('height_range', lambda v: 100 * v.abs().median()), horizontal=('horizontal', lambda v: 100 * v.median()))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.rng, 0.4, color=style.MUTED, label='height from the range alone')
    ax.bar(x + 0.2, g.angle, 0.4, color=style.RADAR, label='height with the angle of arrival')
    for i, r in enumerate(g.itertuples()):
        ax.text(i + 0.2, r.angle + 3, f'{r.angle:.0f}', ha='center', va='bottom', fontsize=8, color=style.INK_2)
        ax.text(i - 0.2, r.rng + 3, f'{r.rng:.0f}', ha='center', va='bottom', fontsize=8, color=style.INK_2)
    ax.set_xticks(x, [f'{iv.left:g}–{iv.right:g}' for iv in g.index])
    ax.set_xlabel('distance of the MoCap joint from the radar axis (m)')
    ax.set_ylabel('median absolute height error (cm)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left')
    ax.set_title(f'Height of the nearest moving reflection against the nearest joint ({len(errors):,} frames)')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(frames=len(errors), horizontal_cm=100 * errors.horizontal.median(),
                angle_cm=100 * errors.height_angle.abs().median(), range_cm=100 * errors.height_range.abs().median(),
                by_bin=g.round(0).to_dict('index'))


def held_posture(posture, hold):
    """the posture with every run shorter than `hold` samples replaced by the preceding stable posture (what the
    transitions count), NaN before the first stable posture"""
    from ..transitions import stable_segments
    out = np.full(len(posture), np.nan)
    for s, e, p in stable_segments(posture, hold):
        out[s:e] = p
    return pd.Series(out).ffill().values


def timeline_figure(t, radar, mocap, h_top, h_med, head, window, events, path):
    """one recording, per radar frame: t (s), radar and MoCap posture codes (MoCap held), radar upper and median
    heights, MoCap head height, evaluation window; events: events.csv rows of the recording"""
    style.apply()
    tm = (t - t[0]) / 60
    radar, mocap = np.where(window, radar, np.nan), np.where(window, mocap, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1.5, 2]))
    ax = axes[0]
    ax.step(tm, 3 - mocap + 0.07, where='post', color=style.MOCAP, lw=1.4, label='MoCap')
    ax.step(tm, 3 - radar - 0.07, where='post', color=style.RADAR, lw=1.4, label='radar')
    for src, colour, y in [('mocap', style.MOCAP, 3.75), ('radar', style.RADAR, 3.45)]:
        for e in events[events.source == src].itertuples():
            ax.plot(e.t_s / 60, y, 'v' if e.direction > 0 else '^', color=colour, ms=5,
                    mfc=colour if e.matched else style.SURFACE, mew=1)
    ax.set_yticks([3, 2, 1, 0], ['standing', 'sitting', 'lying', 'floor'])
    ax.set_ylim(-0.4, 4.0)
    ax.set_title('Posture and transitions in one recording')
    ax = axes[1]
    ok = window & ~np.isnan(h_top)
    ax.plot(tm[ok], h_top[ok], '.', ms=1.5, color=style.RADAR)
    ax.plot(tm[ok], h_med[ok], '.', ms=1.5, color=style.SLOTS[2])
    ax.plot(tm, np.where(window, head, np.nan), color=style.MOCAP, lw=1)
    ax.set_ylabel('height (m)')
    ax.set_xlabel('time (min)')
    handles = [plt.Line2D([], [], color=style.MOCAP, lw=1.4, label='MoCap: posture; head height'),
               plt.Line2D([], [], color=style.RADAR, lw=1.4, label='radar: posture; upper height'),
               plt.Line2D([], [], ls='none', marker='o', ms=4, color=style.SLOTS[2], label='radar: median height'),
               plt.Line2D([], [], ls='none', marker='v', ms=5, color=style.INK_2, label='transition down (▲ up)'),
               plt.Line2D([], [], ls='none', marker='v', ms=5, color=style.INK_2, mfc=style.SURFACE,
                          label='transition without a match')]
    fig.legend(handles=handles, ncol=5, loc='outside lower center')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    m = events[events.source == 'mocap']
    return dict(mocap_transitions=len(m), matched=int(m.matched.sum()),
                radar_transitions=int((events.source == 'radar').sum()))


def posture_figure(confusion, scores, path):
    """confusion: posture_confusion.csv; scores: posture_scores.csv of the category3 run"""
    style.apply()
    cm = confusion.loc[POSTURE_NAMES, POSTURE_NAMES]
    cmn = cm.div(cm.sum(1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axes[0]
    ax.imshow(cmn.values, cmap=LinearSegmentedColormap.from_list('blues', [style.SURFACE] + style.BLUES), vmin=0,
              vmax=1)
    for i in range(4):
        for j in range(4):
            v = cmn.values[i, j]
            ax.text(j, i, f'{v:.1%}', ha='center', va='center', fontsize=9,
                    color=style.SURFACE if v > 0.6 else style.INK)
    ax.set_xticks(range(4), POSTURE_NAMES)
    ax.set_yticks(range(4), POSTURE_NAMES)
    ax.set_xlabel('radar posture')
    ax.set_ylabel('MoCap posture')
    ax.grid(False)
    ax.set_title(f'Posture per epoch, pooled ({int(cm.values.sum()):,} epochs)')
    ax = axes[1]
    rng = np.random.default_rng(0)
    out = dict(diagonal=dict(zip(POSTURE_NAMES, np.diag(cmn.values))), accuracy=np.trace(cm.values) / cm.values.sum())
    xmin = 0.5
    for i, p in enumerate(POSTURE_NAMES):
        v = scores[f'f1_{p}'].dropna()
        y = i + rng.uniform(-0.15, 0.15, len(v))
        low = (v < xmin).values
        ax.plot(v[~low], y[~low], 'o', ms=3, color=style.RADAR, alpha=0.6, mec='none')
        if low.any():                                   # off the axis: drawn on its edge, with their number
            ax.plot(np.full(low.sum(), xmin + 0.005), y[low], '<', ms=5, color=style.RADAR)
            ax.text(xmin + 0.015, i - 0.22, f'{low.sum()} below {xmin:g}', fontsize=7, color=style.INK_2)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=2)
        ax.text(v.median(), i + 0.33, f'median {v.median():.2f}', ha='center', va='top', fontsize=8, color=style.INK_2)
        out[f'f1_{p}'] = (v.median(), v.min())
    ax.set_yticks(range(4), POSTURE_NAMES)
    ax.set_ylim(3.6, -0.6)
    ax.set_xlim(xmin, 1.01)
    ax.set_xlabel('F1 score per recording')
    ax.grid(axis='y', visible=False)
    ax.set_title('Posture F1 per recording (one dot each)')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


def posture_time_figure(mt, path):
    """mt: metrics.csv of the category3 run (posture times in min)"""
    style.apply()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    out = {}
    for ax, p in zip(axes, POSTURE_NAMES):
        g = mt[mt.metric == f'{p}_time']
        d = 60 * (g.radar - g.mocap)
        out[p] = (d.abs().mean(), d.mean(), 60 * g.mocap.median())
        _identity_scatter(ax, g.mocap, g.radar, f'mean absolute error {d.abs().mean():.1f} s\nbias {d.mean():+.1f} s')
        ax.set_xlabel('MoCap (min)')
        ax.set_ylabel('radar (min)')
        ax.set_title('Time ' + (p if p != 'floor' else 'on the floor'))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


def transitions_figure(per_kind, events, path):
    """per_kind: per_kind.csv; events: events.csv of the category3 run"""
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.5, 1.2, 1.1]))
    out = {}
    ax = axes[0]
    k = per_kind.set_index('kind').loc[[c for c, _ in KINDS]]
    y = np.arange(len(k))
    ax.barh(y, k.f1, 0.6, color=style.RADAR)
    for i, r in enumerate(k.itertuples()):
        ax.text(r.f1 + 0.008, i, f'{r.f1:.2f}  ({r.matched} of {r.mocap})', va='center', fontsize=8, color=style.INK_2)
    ax.set_yticks(y, [n for _, n in KINDS])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.12)
    ax.set_xlabel('F1 score (same type, within the tolerance)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Per type, pooled (matched of MoCap)')
    out['f1'] = k.f1.round(3).to_dict()
    ax = axes[1]
    m = events[(events.source == 'radar') & events.matched]
    ax.hist(m.dt_s.clip(-1, 1), bins=np.arange(-1, 1.01, 0.1), color=style.RADAR, edgecolor=style.SURFACE)
    ax.set_xlabel('radar − MoCap time of the transition (s)')
    ax.set_ylabel('matched transitions')
    ax.grid(axis='x', visible=False)
    ax.set_title(f'Timing: median absolute error {m.dt_s.abs().median():.2f} s')
    out['timing_median_s'] = m.dt_s.abs().median()
    ax = axes[2]
    mo = events[events.source == 'mocap']
    g = mo.groupby(pd.cut(mo.offaxis_m, OFFAXIS_BINS), observed=True).agg(n=('matched', 'size'),
                                                                         recall=('matched', 'mean'))
    x = np.arange(len(g))
    ax.bar(x, g.recall, 0.6, color=style.RADAR)
    for i, r in enumerate(g.itertuples()):
        ax.text(i, r.recall + 0.01, f'{r.recall:.2f}\nn={r.n}', ha='center', va='bottom', fontsize=8,
                color=style.INK_2)
    ax.set_xticks(x, [f'{iv.left:g}–{iv.right:g}' for iv in g.index])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel('distance from the radar axis (m)')
    ax.set_ylabel('share of MoCap transitions found')
    ax.grid(axis='x', visible=False)
    ax.set_title('By distance from the radar axis')
    out['recall_by_offaxis'] = g.recall.round(2).tolist()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out


def matched_peaks(events):
    """per matched transition of a counted type: the MoCap and the radar peak radial speed (m/s)"""
    mo = events[(events.source == 'mocap') & events.matched & events.kind.isin([c for c, _ in COUNTED])]
    ra = events[(events.source == 'radar') & events.matched]
    rows = []
    for r in mo.itertuples():
        c = ra[(ra.rec == r.rec) & (ra.kind == r.kind) & ((ra.t_s - (r.t_s - r.dt_s)).abs() < 0.01)]
        if len(c):
            rows.append(dict(rec=r.rec, kind=r.kind, mocap=r.peak_v, radar=c.peak_v.iloc[0]))
    return pd.DataFrame(rows).dropna()


def peak_figure(events, mt, path, axis_max=3.0):
    """events: events.csv; mt: metrics.csv of the category3 run"""
    style.apply()
    pp = matched_peaks(events)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, axis_max], [0, axis_max], color=style.INK_2, lw=1, zorder=1)
    for (kind, name), colour in zip(COUNTED, style.SLOTS):
        q = pp[pp.kind == kind]
        ax.scatter(q.mocap.clip(upper=axis_max), q.radar.clip(upper=axis_max), s=8, color=colour, alpha=0.6,
                   edgecolor='none', label=name, zorder=2)
    d = pp.radar - pp.mocap
    outside = int(((pp.mocap > axis_max) | (pp.radar > axis_max)).sum())
    ax.text(0.04, 0.96, f'{len(pp):,} matched transitions\nmedian absolute error {d.abs().median():.2f} m/s\n'
                        f'median bias {d.median():+.2f} m/s', transform=ax.transAxes, va='top', fontsize=8,
            color=style.INK_2)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect('equal')
    ax.set_xlabel(f'MoCap peak radial speed (m/s; {outside} values above {axis_max:g} on the edge)')
    ax.set_ylabel('radar peak radial speed (m/s)')
    ax.legend(loc='lower right')
    ax.set_title('Per transition')
    ax = axes[1]
    g = mt[mt.metric == 'peak_transition_doppler']
    d2 = g.radar - g.mocap
    _identity_scatter(ax, g.mocap, g.radar, f'mean absolute error {d2.abs().mean():.2f} m/s\nbias {d2.mean():+.2f} m/s',
                      lo=0.5)
    ax.set_xlabel('MoCap (m/s)')
    ax.set_ylabel('radar (m/s)')
    ax.set_title('Mean peak speed per recording (the metric)')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(transitions=len(pp), transition_mae_median=d.abs().median(), transition_bias_median=d.median(),
                recording_mae=d2.abs().mean(), recording_bias=d2.mean(), outside=outside)


def per_participant_table(ps, es, mt, person_of):
    """per participant: mean posture F1 over their recordings, transition F1 pooled, mean absolute error of each
    posture time (s) and of the mean peak speed (m/s) over their recordings"""
    ps, es, mt = (d.assign(person=person_of(d.rec)) for d in (ps, es, mt))
    rows = []
    for p, g in es.groupby('person'):
        row = dict(person=p, posture_f1=ps[ps.person == p].f1_mean.mean(),
                   transition_f1=2 * g.matched.sum() / (g.mocap_events.sum() + g.radar_events.sum()))
        for post in POSTURE_NAMES:
            h = mt[(mt.person == p) & (mt.metric == f'{post}_time')]
            row[post] = 60 * (h.radar - h.mocap).abs().mean()
        h = mt[(mt.person == p) & (mt.metric == 'peak_transition_doppler')]
        row['peak'] = (h.radar - h.mocap).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def per_participant_figure(table, path):
    """table: per_participant_table. One dot per participant (unlabelled); the median as a bar"""
    style.apply()
    rng = np.random.default_rng(0)
    panels = [('Scores', [('posture_f1', 'posture F1'), ('transition_f1', 'transition F1')], 'score', '{:.2f}',
               (0.65, 1.0)),
              ('Posture time', [(p, p if p != 'floor' else 'on the floor') for p in POSTURE_NAMES],
               'mean absolute error (s)', '{:.1f} s', (0, None)),
              ('Peak speed', [('peak', 'mean peak speed')], 'mean absolute error (m/s)', '{:.2f}', (0, None))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1.8, 1]))
    out = {}
    for ax, (title, cols, xlabel, fmt, xlim) in zip(axes, panels):
        for i, (c, label) in enumerate(cols):
            out[c] = (_strip(ax, i, table[c].values, rng, fmt), table[c].min(), table[c].max())
        ax.set_yticks(range(len(cols)), [label for _, label in cols])
        ax.set_ylim(len(cols) - 0.4, -0.6)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', visible=False)
        ax.set_title(title)
    fig.suptitle(f'Per participant ({len(table)} participants, one dot each; bar: median)', x=0.01, ha='left',
                 fontsize=10, color=style.INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return out
