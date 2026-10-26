"""
Activity bouts of the radar and of MoCap over time, per recording (mobility/plots/bouts.py). Uses the thresholds and
bout rules of the Category 2 evaluation: run scripts/evaluate_category2.py first.

    python scripts/plot_bouts.py                                       # every evaluated recording
    python scripts/plot_bouts.py --only person_10/scenario_2 person_15/scenario_4

Writes OUTPUT_ROOT/plots/bouts/<person>_<scenario>_bouts.png
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import BoutRule  # noqa: E402
from mobility.config import EPOCH_S, OUTPUT_ROOT  # noqa: E402
from mobility.evaluation import add_bouts, epoch_pairs  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.bouts import bouts_figure  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    args = ap.parse_args()
    th = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv')).set_index('rec')
    rules = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category2', 'rules.csv')).set_index('rec')
    recs = args.only or list(rules.index)
    out = os.path.join(OUTPUT_ROOT, 'plots', 'bouts')
    os.makedirs(out, exist_ok=True)
    for rec in recs:
        r, b = th.loc[rec], rules.loc[rec]
        rule = BoutRule(int(round(b.max_gap_s / EPOCH_S)), int(round(b.min_bout_s / EPOCH_S)))
        s = load_session(*parse_recording(rec))
        pairs = add_bouts(epoch_pairs(s, r.radar_threshold, r.mocap_threshold, r.active_feature, r.energy_feature),
                          rule)
        path = os.path.join(out, f'{s.person}_{s.scenario}_bouts.png')
        bouts_figure(s, pairs, rule, path)
        print(path)
