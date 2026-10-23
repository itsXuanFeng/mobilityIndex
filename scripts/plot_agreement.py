"""
Radar vs MoCap over time, per recording: the whole recording and close-ups of typical episodes (mobility/plots/
agreement.py). Uses the thresholds of the Category 1 evaluation: run scripts/evaluate_category1.py first.

    python scripts/plot_agreement.py                                   # every evaluated recording
    python scripts/plot_agreement.py --only person_10/scenario_2 person_15/scenario_4

Writes OUTPUT_ROOT/plots/agreement/<person>_<scenario>_recording.png and _episodes.png
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import OUTPUT_ROOT  # noqa: E402
from mobility.evaluation import epoch_pairs  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.agreement import episodes_figure, recording_figure  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    args = ap.parse_args()
    th = pd.read_csv(os.path.join(OUTPUT_ROOT, 'category1', 'thresholds.csv')).set_index('rec')
    recs = [parse_recording(r) for r in args.only] if args.only else [parse_recording(r) for r in th.index]
    out = os.path.join(OUTPUT_ROOT, 'plots', 'agreement')
    os.makedirs(out, exist_ok=True)
    for person, scenario in recs:
        r = th.loc[f'{person}/{scenario}']
        s = load_session(person, scenario)
        pairs = epoch_pairs(s, r.radar_threshold, r.mocap_threshold, r.active_feature, r.energy_feature)
        for kind, fn in (('recording', recording_figure), ('episodes', episodes_figure)):
            path = os.path.join(out, f'{person}_{scenario}_{kind}.png')
            fn(s, pairs, path)
            print(path)
