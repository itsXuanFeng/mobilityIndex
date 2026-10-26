"""
Postural transitions of the radar and of MoCap over time, per recording (mobility/plots/transitions.py). Uses the
transitions of the Category 3 evaluation: run scripts/evaluate_category3.py first.

    python scripts/plot_transitions.py                                       # every evaluated recording
    python scripts/plot_transitions.py --only person_10/scenario_2 person_1/scenario_1

Writes OUTPUT_ROOT/plots/transitions/<person>_<scenario>_transitions.png
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import OUTPUT_ROOT  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.transitions import transitions_figure  # noqa: E402
from mobility.posture import posture_rules, radar_height, radar_offaxis, radar_posture_rules  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    args = ap.parse_args()
    events = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category3', 'events.csv'), keep_default_na=False,
                         na_values=[''])
    rules, radar_rules = posture_rules(), radar_posture_rules()
    out = os.path.join(OUTPUT_ROOT, 'plots', 'transitions')
    os.makedirs(out, exist_ok=True)
    for rec in args.only or list(events.rec.unique()):
        s = load_session(*parse_recording(rec))
        rule, radar_rule = rules.get(s.person, rules['ALL']), radar_rules.get(s.person, radar_rules['ALL'])
        h, off = radar_height(s), radar_offaxis(s)
        path = os.path.join(out, f'{s.person}_{s.scenario}_transitions.png')
        transitions_figure(s, rule.classify(s.mocap), radar_rule.classify(h, off), radar_rule.height(h, off),
                           events[events.rec == rec], path)
        print(path)
