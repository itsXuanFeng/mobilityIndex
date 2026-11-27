"""
Category 2 result figures (mobility/plots/category2.py), from the latest category2 run (bouts.csv, bout_scores.csv,
metrics.csv, epoch_scores.csv, rules.csv) and the category1 thresholds: run scripts/evaluate_category2.py first.
The bout formation figure rebuilds the states of one recording as the evaluation did (as scripts/plot_bouts.py).

    python scripts/plot_category2.py            # to its run folder
    python scripts/plot_category2.py --docs     # also copies the figures to docs/figures/ (public docs)

Writes c2_formation.png, c2_matching.png, c2_metrics.png, c2_per_participant.png and c2_checks.png,
and prints the numbers they show.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.bouts import BoutRule, hysteresis  # noqa: E402
from mobility.config import EPOCH_S, REPO  # noqa: E402
from mobility.evaluation import (bout_frame, paired_states, posture_events, set_bout_levels,  # noqa: E402
                                 set_bout_transitions)
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.category2 import (checks_figure, formation_figure, matching_figure,  # noqa: E402
                                      metrics_figure, per_participant_figure, per_participant_table)
from mobility.posture import posture_rules, radar_posture_models  # noqa: E402
from mobility.session import load_session  # noqa: E402
from mobility.states import distance_corrected  # noqa: E402
from mobility.utils import fold_of, person_of  # noqa: E402

EXAMPLE = 'person_10/scenario_2'    # the bout formation figure's recording (as Category 1's timeline)
DOCS_FIGURES = os.path.join(REPO, 'docs', 'figures')


def formation_inputs(rec):
    """per epoch of one recording, as in the evaluation: radar distance-corrected level, valid, candidate runs (after
    the hysteresis, before the confirmation), radar and MoCap bouts, radar transition times (s since the first epoch),
    (exit, entry, confirmation)"""
    th = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec').loc[rec]
    b = pd.read_csv(latest('category2', 'rules.csv')).set_index('rec').loc[rec]
    s = load_session(*parse_recording(rec))
    rs, ms = paired_states(s, th.radar_threshold, th.mocap_threshold, active_feature=th.active_feature,
                           energy_feature=th.energy_feature)
    radar_ev, mocap_ev, _ = posture_events(s, fold_of(radar_posture_models(), s.person),
                                           fold_of(posture_rules(), s.person), th.radar_threshold, th.active_feature)
    set_bout_transitions(rs, ms, radar_ev.t.values, mocap_ev.t.values)
    rs = distance_corrected(rs, b.radar_distance_slope, b.radar_entry)
    rs.extra['bout_rule'] = ms.extra['bout_rule'] = BoutRule(int(round(b.max_gap_s / EPOCH_S)),
                                                             int(round(b.min_bout_s / EPOCH_S)))
    set_bout_levels(rs, ms, (b.radar_exit, b.radar_confirm), (b.mocap_exit, b.mocap_confirm))
    frame = bout_frame(rs, ms)
    candidates = hysteresis(rs.active, rs.level, rs.valid, b.radar_exit)
    return (rs.level, rs.valid, candidates, frame.radar_bout.values, frame.mocap_bout.values,
            (radar_ev.t.values - rs.t[0]) / EPOCH_S, (b.radar_exit, b.radar_entry, b.radar_confirm))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs', action='store_true', help='copy the figures to docs/figures/')
    ap.add_argument('--example', default=EXAMPLE, help='recording of the bout formation figure')
    ap.add_argument('--start', type=int, help='first epoch of its segment (default: chosen automatically)')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_category2', args, partial=False) as run_dir:
        bouts = pd.read_csv(latest('category2', 'bouts.csv'))
        scores = pd.read_csv(latest('category2', 'bout_scores.csv'))
        mt = pd.read_csv(latest('category2', 'metrics.csv'))
        eps = pd.read_csv(latest('category2', 'epoch_scores.csv'))
        names = ('formation', 'matching', 'metrics', 'per_participant', 'checks')
        paths = {k: os.path.join(run_dir, f'c2_{k}.png') for k in names}
        start = formation_figure(*formation_inputs(args.example), paths['formation'], start=args.start)
        print(f'formation: {args.example}, segment from epoch {start}')
        print('matching:', matching_figure(bouts, scores, paths['matching']))
        print('metrics (median |error| %, mean bias %):', metrics_figure(mt, paths['metrics']))
        table = per_participant_table(mt, scores, person_of)
        print('per participant (medians):', per_participant_figure(table, paths['per_participant']))
        print('checks:', checks_figure(eps, bouts, paths['checks']))
        print(f'{bouts.rec.nunique()} recordings, {table.person.nunique()} participants; '
              f'{np.mean(scores.bout_f1 >= 0.9):.0%} of recordings with bout F1 >= 0.9')
        if args.docs:
            os.makedirs(DOCS_FIGURES, exist_ok=True)
            for p in paths.values():
                shutil.copy(p, DOCS_FIGURES)
            print(f'copied to {DOCS_FIGURES}')
