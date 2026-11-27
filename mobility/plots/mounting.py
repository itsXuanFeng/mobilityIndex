"""
The mounting check from the walking height (scripts/check_mounting_height.py): left, the walking height against the
horizontal distance from the radar axis and the distance curve that corrects it; middle, each person's corrected
walking-height offset (all their recordings) with the check band of the other persons; right, the share of radars
flagged per simulated error of the installed height (the exact-stature variant is in detection.csv only).
"""
import matplotlib.pyplot as plt
import numpy as np

from . import style

# three series at most: the first three categorical slots are the ones that stay apart for every pair
SERIES = (('recording', 'walking height', 'one recording (~10 min)'),
          ('all', 'walking height', 'all recordings (~1 h)'),
          ('all', 'with stature +-2 cm', 'all recordings + stature typed in (±2 cm)'))
LABEL_AT = 0.10    # m: the error where the direct labels sit (the curves are apart there)


def plot_mounting_check(persons, check_all, detection, by_distance, curve, path):
    """persons: persons.csv; check_all: mounting_check.csv rows of the held-out persons with data 'all' and the
    distance curve; detection: detection.csv rows with the distance curve; by_distance: by_distance.csv; curve: the
    DistanceCurve of all persons"""
    style.apply()
    fig, (z, a, b) = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(width_ratios=(0.9, 1.2, 1)))

    bd = by_distance[by_distance.frames >= 200]
    mid = (bd.d_lo + bd.d_hi) / 2
    z.fill_between(mid, 100 * bd.q25, 100 * bd.q75, color=style.GRID, alpha=0.7, lw=0,
                   label='middle half of the walking frames')
    z.plot(mid, 100 * bd['median'], 'o', ms=6, color=style.SLOTS[0], mec=style.SURFACE, mew=1.5,
           label='median per 0.25 m')
    dd = np.linspace(bd.d_lo.min(), bd.d_hi.max(), 200)
    z.plot(dd, 100 * curve(dd), '-', color=style.INK_2, lw=1.4, label='distance curve (subtracted)')
    z.axhline(0, color=style.BASELINE, lw=0.75)
    z.set_xlabel('horizontal distance from the radar axis (m)')
    z.set_ylabel("walking top-of-body height, minus the\nstanding line and the person's median (cm)")
    z.set_title('Walking height against distance')
    z.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)

    p = persons.sort_values('offset_curve').reset_index(drop=True)
    c = check_all.set_index('person').loc[p.person]
    x = np.arange(len(p))
    a.fill_between(x, 100 * (c.reference - c.limit), 100 * (c.reference + c.limit), color=style.GRID, alpha=0.6,
                   step='mid', lw=0, label='plausible: within the band of the other persons (walking height alone)')
    a.axhline(0, color=style.BASELINE, lw=0.75)
    a.plot(x, 100 * p.offset_curve, 'o', ms=6, color=style.SLOTS[0], mec=style.SURFACE, mew=1.5,
           label="the person's walking-height offset (all recordings)")
    a.set_xticks(x, [s.replace('person_', 'p') for s in p.person], rotation=90, fontsize=7)
    a.set_ylabel('top of the body while walking, minus the\npopulation standing height (cm)')
    a.set_title('Walking height per person: mostly stature')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)

    for k, (data, variant, label) in enumerate(SERIES):
        g = detection[(detection.data == data) & (detection.variant == variant)].sort_values('delta')
        b.plot(100 * g.delta, 100 * g.flagged, '-o', ms=5, color=style.SLOTS[k], mec=style.SURFACE, mew=1.2,
               label=label)
        y = 100 * g.flagged[np.isclose(g.delta, LABEL_AT)].iloc[0]
        b.annotate(label.split(' (')[0] if k < 2 else 'with stature', (100 * LABEL_AT, y),
                   xytext=(7, -9 if y > 95 else -2),
                   textcoords='offset points', fontsize=7, color=style.INK_2, va='center')
    b.axvline(0, color=style.BASELINE, lw=0.75)
    b.set_xlabel('error of the installed radar height (cm)')
    b.set_ylabel('radars flagged (%)')
    b.set_ylim(-3, 103)
    b.set_title('Errors caught (at 0 cm: false alarms)')
    b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
