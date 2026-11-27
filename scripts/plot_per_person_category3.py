"""
Category 3 per person: signed error radar - MoCap per recording of every Category 3 metric: the sit-to-stand,
stand-to-sit and lie-to-sit counts, the peak transition speed and the time standing, sitting, lying and on the floor.
Two figures:
    per_person_category3_bias.png       in the metric's own units (transitions, m/s, min)
    per_person_category3_bias_pct.png   in % of the MoCap value, 100 x (radar - MoCap) / MoCap; recordings where MoCap
                                        is 0 (no floor time, no lie-to-sit) have no % and are left out of that panel
Bars: the person's mean over their recordings (red: radar too high, blue: too low); dots: one recording each; 'all':
the mean over all recordings, whisker 10th-90th percentile. All on one side = a systematic error for that person.
Each panel has its own y-range, fitted to its data.
Reads metrics.csv of the latest category3 run: run scripts/evaluate_category3.py first.

    python scripts/plot_per_person_category3.py

Writes both figures to its run folder
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.plots.per_person import CATEGORY3_PANELS, signed_error_figure  # noqa: E402
from mobility.utils import person_of  # noqa: E402


def signed_errors(percent=False):
    """per recording and metric: person, radar - MoCap in the metric's units, or (percent) in % of MoCap, NaN where
    MoCap is 0"""
    mt = pd.read_csv(latest('category3', 'metrics.csv'))
    d = mt.radar - mt.mocap
    err = 100 * d / mt.mocap.where(mt.mocap != 0, np.nan) if percent else d
    return mt.assign(person=person_of(mt.rec), err=err)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    args = ap.parse_args()
    keys = [k for k, _, _ in CATEGORY3_PANELS]
    with run('plot_per_person_category3', args, partial=False) as run_dir:
        for percent, name, panels in ((False, 'per_person_category3_bias.png', CATEGORY3_PANELS),
                                      (True, 'per_person_category3_bias_pct.png',
                                       [(k, t, '%') for k, t, _ in CATEGORY3_PANELS])):
            e = signed_errors(percent)
            persons = sorted(e.person.unique(), key=natural_key)
            path = os.path.join(run_dir, name)
            signed_error_figure(e, persons, panels, path, 'Category 3' + (' (% of MoCap)' if percent else ''))
            print(f'wrote {path}')
            t = e.groupby(['metric', 'person']).err.mean().unstack('metric')[keys]
            print(t.reindex(persons).round(1 if percent else 2).to_string())
