"""
Activity bouts of the radar and of MoCap over time, per recording (mobility/plots/bouts.py). Uses the thresholds and
bout definitions of the Category 2 evaluation: run scripts/evaluate_category2.py first.

    python scripts/plot_bouts.py                                       # every evaluated recording
    python scripts/plot_bouts.py --only person_10/scenario_2 person_15/scenario_4

Writes <person>_<scenario>_bouts.png to its run folder
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import BoutRule  # noqa: E402
from mobility.config import EPOCH_S  # noqa: E402
from mobility.evaluation import bout_frame, paired_states, set_bout_levels  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.bouts import bouts_figure  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_bouts', args, partial=False) as run_dir:
        th = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec')
        rules = pd.read_csv(latest('category2', 'rules.csv')).set_index('rec')
        recs = args.only or list(rules.index)
        out = run_dir
        for rec in recs:
            r, b = th.loc[rec], rules.loc[rec]
            s = load_session(*parse_recording(rec))
            rs, ms = paired_states(s, r.radar_threshold, r.mocap_threshold, active_feature=r.active_feature,
                                   energy_feature=r.energy_feature)
            rule = BoutRule(int(round(b.max_gap_s / EPOCH_S)), int(round(b.min_bout_s / EPOCH_S)))
            rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
            set_bout_levels(rs, ms, (b.radar_exit, b.radar_confirm), (b.mocap_exit, b.mocap_confirm))
            note = (f'MoCap exit {b.mocap_exit_factor:g}x, confirm {b.mocap_confirm_factor:g}x entry; radar exit '
                    f'-{b.radar_exit_delta:g}, confirm +{b.radar_confirm_eps:g}; {rule}. ')
            path = os.path.join(out, f'{s.person}_{s.scenario}_bouts.png')
            bouts_figure(s, bout_frame(rs, ms), path, note)
            print(path)
