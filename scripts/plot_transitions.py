"""
Postural transitions of the radar and of MoCap over time, per recording (mobility/plots/transitions.py). Uses the
transitions of the Category 3 evaluation: run scripts/evaluate_category3.py first.

    python scripts/plot_transitions.py                                       # every evaluated recording
    python scripts/plot_transitions.py --only person_10/scenario_2 person_1/scenario_1

Writes <person>_<scenario>_transitions.png to its run folder
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.transitions import transitions_figure  # noqa: E402
from mobility.posture import (decode_postures, posture_rules, radar_height, radar_median_height,  # noqa: E402
                              radar_moving, radar_offaxis, radar_posture_models)
from mobility.session import load_session  # noqa: E402
from mobility.utils import fold_of  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_transitions', args, partial=False) as run_dir:
        events = pd.read_csv(latest('category3', 'events.csv'), keep_default_na=False,
                             na_values=[''])
        th = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec')    # the movement gate
        rules, models = posture_rules(), radar_posture_models()
        out = run_dir
        for rec in args.only or list(events.rec.unique()):
            s = load_session(*parse_recording(rec))
            rule, model = fold_of(rules, s.person), fold_of(models, s.person)
            h = radar_height(s)
            moving = radar_moving(s, th.loc[rec, 'radar_threshold'], th.loc[rec, 'active_feature'])
            radar = decode_postures(model, [(h, radar_median_height(s), radar_offaxis(s), moving)])[0]
            path = os.path.join(out, f'{s.person}_{s.scenario}_transitions.png')
            transitions_figure(s, rule.classify(s.mocap), radar, h, events[events.rec == rec], path)
            print(path)
