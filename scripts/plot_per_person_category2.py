"""
Category 2 per person: signed error in % of the MoCap value, 100 x (radar - MoCap) / MoCap per recording, of the four
metrics: number of bouts, mean and variance of the bout duration, fragmentation index. Bars: the person's mean over
their recordings (red: radar too high, blue: too low); dots: one recording each; 'all': the mean over all recordings,
whisker 10th-90th percentile. All on one side = a systematic error for that person. Each panel has its own
y-range, fitted to its data.
Reads metrics.csv of the latest category2 run: run scripts/evaluate_category2.py first.

    python scripts/plot_per_person_category2.py

Writes per_person_bouts_bias.png to its run folder
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.plots.per_person import BOUT_PANELS, bouts_bias_figure  # noqa: E402
from mobility.utils import person_of  # noqa: E402


def signed_errors():
    """per recording and metric: person, radar - MoCap in % of MoCap"""
    mt = pd.read_csv(latest('category2', 'metrics.csv'))
    return mt.assign(person=person_of(mt.rec), err=100 * (mt.radar - mt.mocap) / mt.mocap)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_per_person_category2', args, partial=False) as run_dir:
        e = signed_errors()
        persons = sorted(e.person.unique(), key=natural_key)
        path = os.path.join(run_dir, 'per_person_bouts_bias.png')
        bouts_bias_figure(e, persons, path)
        print(f'wrote {path}')
        t = e.groupby(['metric', 'person']).err.mean().unstack('metric')[[k for k, _ in BOUT_PANELS]]
        print(t.reindex(persons).round(1).to_string())
