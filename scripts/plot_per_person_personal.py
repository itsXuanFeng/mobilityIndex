"""
Personal calibration per person: each person's mean signed error, 100 x (radar - MoCap) / MoCap over their recordings,
under setups A (LOSO), B (LOSO + clip) and C (LOSO + other recordings + clip), one figure per category.
Reads metrics.csv of the latest personal run: run scripts/evaluate_personal_calibration.py first. The energy is the
radar energy after the setup's conversion factor; the SD of the bout duration is the square root of its variance. The
floor time is left out: its MoCap value is often near 0, so its % error is not meaningful.

    python scripts/plot_per_person_personal.py

Writes personal_category{1,2,3}_bias.png to its run folder
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.plots.per_person import setups_bias_figure  # noqa: E402

FIGURES = {  # category: panels (metric, title)
    1: [('active_time', 'Active time'), ('inactive_time', 'Inactive time'),
        ('total_activity_energy', 'Total activity energy')],
    2: [('n_bouts', 'Number of bouts'), ('bout_duration_mean', 'Mean bout duration'),
        ('bout_duration_sd', 'SD of bout duration'), ('fragmentation_index', 'Fragmentation index')],
    3: [('standing_time', 'Standing time'), ('sitting_time', 'Sitting time'), ('lying_time', 'Lying time')],
}


def signed_errors():
    """per recording, setup and metric: person, radar - MoCap in % of MoCap (finite values only)"""
    mt = pd.read_csv(latest('personal', 'metrics.csv'))
    sd = mt[mt.metric == 'bout_duration_var'].assign(metric='bout_duration_sd', radar=lambda d: np.sqrt(d.radar),
                                                     mocap=lambda d: np.sqrt(d.mocap))
    mt = pd.concat([mt, sd], ignore_index=True)
    mt['err'] = 100 * (mt.radar - mt.mocap) / mt.mocap
    bad = ~np.isfinite(mt.err)
    if bad.any():
        print(f'left out, MoCap value 0 or missing: {mt[bad].groupby("metric").size().to_dict()}')
    return mt[~bad]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_per_person_personal', args, partial=False) as run_dir:
        e = signed_errors()
        persons = sorted(e.person.unique(), key=natural_key)
        OUT = run_dir
        n_rec, n_p = e.rec.nunique(), len(persons)
        for cat, panels in FIGURES.items():
            path = os.path.join(OUT, f'personal_category{cat}_bias.png')
            title = (f'Category {cat}, personal calibration: signed error per person '
                     f'({n_rec} recordings, {n_p} people; recording minus the clip)')
            setups_bias_figure(e, persons, panels, path, title)
            print(f'wrote {path}')
            t = e[e.metric.isin([k for k, _ in panels])].groupby(['metric', 'setup']).err
            print(pd.DataFrame(dict(mae_pct=t.apply(lambda x: x.abs().mean()), bias_pct=t.mean())).round(1).to_string())
