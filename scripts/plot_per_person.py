"""
Category 1 per person, as bar plots: AUC and F1 of the active / inactive calls, and the error of the total activity
energy. Reads per_person.csv and agreement.csv of the latest category1 run: run scripts/evaluate_category1.py first.

    python scripts/plot_per_person.py

Writes to its run folder per_person_scores.png, per_person_energy_error.png and per_person_energy_bias.png
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.evaluation import energy_conversion  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.plots.per_person import energy_bias_figure, energy_error_figure, scores_figure  # noqa: E402


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_per_person', args, partial=False) as run_dir:
        pp = pd.read_csv(latest('category1', 'per_person.csv'))
        ag = pd.read_csv(latest('category1', 'agreement.csv')).set_index('metric')
        OUT = run_dir
        scores_figure(pp, os.path.join(OUT, 'per_person_scores.png'))
        energy_error_figure(pp, ag.loc['total_activity_energy', 'baseline_error_pct'],
                            os.path.join(OUT, 'per_person_energy_error.png'))
        mt = pd.read_csv(latest('category1', 'metrics.csv'))
        c = energy_conversion(mt[mt.metric == 'total_activity_energy'])
        c['bias_pct'] = 100 * (c.converted - c.mocap) / c.mocap
        energy_bias_figure(pp, c, os.path.join(OUT, 'per_person_energy_bias.png'))
        print(f'wrote {OUT}')
