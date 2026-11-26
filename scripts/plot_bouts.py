"""
Activity bouts of the radar and of MoCap over time, per recording (mobility/plots/bouts.py). Uses the thresholds and
bout definitions of the Category 2 evaluation (distance correction, entry, kappa, MoCap factors, bout rule) and the
posture transitions of the posture models it used: run scripts/evaluate_category2.py first.

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
from mobility.evaluation import (bout_frame, paired_states, posture_events, set_bout_levels,  # noqa: E402
                                 set_bout_transitions)
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.bouts import bouts_figure  # noqa: E402
from mobility.posture import posture_rules, radar_posture_models  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.states import distance_corrected  # noqa: E402
from mobility.utils import fold_of  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_bouts', args, partial=False) as run_dir:
        th = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec')
        rules = pd.read_csv(latest('category2', 'rules.csv')).set_index('rec')
        recs = args.only or list(rules.index)
        models, mocap_rules = radar_posture_models(), posture_rules()
        out = run_dir
        for rec in recs:
            r, b = th.loc[rec], rules.loc[rec]
            s = load_session(*parse_recording(rec))
            rs, ms = paired_states(s, r.radar_threshold, r.mocap_threshold, active_feature=r.active_feature,
                                   energy_feature=r.energy_feature)
            radar_ev, mocap_ev, _ = posture_events(s, fold_of(models, s.person), fold_of(mocap_rules, s.person),
                                                   r.radar_threshold, r.active_feature)
            set_bout_transitions(rs, ms, radar_ev.t.values, mocap_ev.t.values)
            rs = distance_corrected(rs, b.radar_distance_slope, b.radar_entry)
            rule = BoutRule(int(round(b.max_gap_s / EPOCH_S)), int(round(b.min_bout_s / EPOCH_S)))
            rs.extra['bout_rule'] = ms.extra['bout_rule'] = rule
            speed = b.get('radar_confirm_speed')                 # absent in runs before 2026-11-22
            speed = None if speed is None or pd.isna(speed) else float(speed)
            set_bout_levels(rs, ms, (b.radar_exit, b.radar_confirm), (b.mocap_exit, b.mocap_confirm), speed)
            note = (f'MoCap exit / confirm {b.mocap_exit_factor:g}x / {b.mocap_confirm_factor:g}x entry; radar margin '
                    f'{b.radar_kappa:g}, distance-corrected{"" if speed is None else f", {speed:g} m/s"}; '
                    f'transitions confirm; {rule}. ')
            path = os.path.join(out, f'{s.person}_{s.scenario}_bouts.png')
            bouts_figure(s, bout_frame(rs, ms), path, note)
            print(path)
