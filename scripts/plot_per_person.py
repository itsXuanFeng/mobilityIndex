"""
Category 1 per person, as bar plots: AUC and F1 of the active / inactive calls, and the error of the total activity
energy. Reads OUTPUT_ROOT/category1/per_person.csv and agreement.csv: run scripts/evaluate_category1.py first.

    python scripts/plot_per_person.py

Writes OUTPUT_ROOT/plots/per_person/per_person_scores.png, per_person_energy_error.png and per_person_energy_bias.png
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import PLOTS_DIR, category_dir  # noqa: E402
from mobility.evaluation import energy_conversion  # noqa: E402
from mobility.plots.per_person import energy_bias_figure, energy_error_figure, scores_figure  # noqa: E402

IN = category_dir(1)
OUT = os.path.join(PLOTS_DIR, 'per_person')

if __name__ == '__main__':
    pp = pd.read_csv(os.path.join(IN, 'per_person.csv'))
    ag = pd.read_csv(os.path.join(IN, 'agreement.csv')).set_index('metric')
    os.makedirs(OUT, exist_ok=True)
    scores_figure(pp, os.path.join(OUT, 'per_person_scores.png'))
    energy_error_figure(pp, ag.loc['total_activity_energy', 'baseline_error_pct'],
                        os.path.join(OUT, 'per_person_energy_error.png'))
    mt = pd.read_csv(os.path.join(IN, 'metrics.csv'))
    c = energy_conversion(mt[mt.metric == 'total_activity_energy'])
    c['bias_pct'] = 100 * (c.converted - c.mocap) / c.mocap
    energy_bias_figure(pp, c, os.path.join(OUT, 'per_person_energy_bias.png'))
    print(f'wrote {OUT}')
