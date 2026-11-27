"""
Category 1 result figures (mobility/plots/category1.py), from the latest category1 run (epochs.csv.gz, metrics.csv,
thresholds.csv: run scripts/evaluate_category1.py first):

    python scripts/plot_category1.py            # to its run folder
    python scripts/plot_category1.py --docs     # also copies the figures to docs/figures/ (public docs)

Writes c1_timeline.png, c1_roc.png, c1_agreement.png and c1_per_participant.png, and prints the numbers they show.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import REPO  # noqa: E402
from mobility.evaluation.agreement import energy_conversion  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.plots.category1 import (agreement_figure, per_participant_figure, per_participant_table,  # noqa: E402
                                      roc_figure, timeline_figure)
from mobility.utils import person_of  # noqa: E402

EXAMPLE = 'person_10/scenario_2'    # the timeline's recording: a chair far off the radar axis, a typical F1
DOCS_FIGURES = os.path.join(REPO, 'docs', 'figures')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs', action='store_true', help='copy the figures to docs/figures/')
    ap.add_argument('--example', default=EXAMPLE, help='recording of the timeline figure')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_category1', args, partial=False) as run_dir:
        ep = pd.read_csv(latest('category1', 'epochs.csv.gz'))
        mt = pd.read_csv(latest('category1', 'metrics.csv')).assign(person=lambda d: person_of(d.rec))
        thr = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec')
        times = mt[mt.metric == 'active_time']
        e_rows = mt[mt.metric == 'total_activity_energy']
        energy = energy_conversion(e_rows).assign(radar=e_rows.radar.values)
        paths = {k: os.path.join(run_dir, f'c1_{k}.png') for k in ('timeline', 'roc', 'agreement', 'per_participant')}
        print('timeline:', timeline_figure(ep, args.example, thr.loc[args.example, 'mocap_threshold'],
                                           paths['timeline']))
        print('ROC:', roc_figure(ep, paths['roc']))
        print('agreement:', agreement_figure(times, energy, paths['agreement']))
        table = per_participant_table(ep, times, energy)
        print('per participant (median, min, max):', per_participant_figure(table, paths['per_participant']))
        if args.docs:
            os.makedirs(DOCS_FIGURES, exist_ok=True)
            for p in paths.values():
                shutil.copy(p, DOCS_FIGURES)
            print(f'copied to {DOCS_FIGURES}')
