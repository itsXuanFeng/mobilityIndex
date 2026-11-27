"""
Category 3 result figures (mobility/plots/category3.py), from the latest category3 run (events.csv, metrics.csv,
posture_confusion.csv, posture_scores.csv, event_scores.csv, per_kind.csv) and the category1 thresholds (the movement
gate of the example recording).
The timeline rebuilds the postures of one recording as scripts/plot_transitions.py does.

    python scripts/plot_category3.py            # to its run folder
    python scripts/plot_category3.py --docs     # also copies the figures to docs/figures/ (public docs)

Writes c3_timeline.png, c3_posture.png, c3_posture_time.png, c3_transitions.png, c3_peak.png and
c3_per_participant.png, and prints the numbers they show.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import FPS, REPO  # noqa: E402
from mobility.experiment import add_run_args, latest, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.category3 import (held_posture, peak_figure, per_participant_figure,  # noqa: E402
                                      per_participant_table, posture_figure, posture_time_figure, timeline_figure,
                                      transitions_figure)
from mobility.posture import (decode_postures, posture_rules, radar_height, radar_median_height,  # noqa: E402
                              radar_moving, radar_offaxis, radar_posture_models)
from mobility.session import load_session  # noqa: E402
from mobility.utils import fold_of, person_of  # noqa: E402

EXAMPLE = 'person_10/scenario_2'    # the timeline's recording (as in Categories 1 and 2)
DOCS_FIGURES = os.path.join(REPO, 'docs', 'figures')


def timeline_inputs(rec, events):
    """per radar frame of one recording, as in the evaluation: t, radar posture (posture model), MoCap posture (rule,
    held as its transitions count it), radar upper and median heights, MoCap head height, evaluation window"""
    th = pd.read_csv(latest('category1', 'thresholds.csv')).set_index('rec').loc[rec]
    s = load_session(*parse_recording(rec))
    rule, model = fold_of(posture_rules(), s.person), fold_of(radar_posture_models(), s.person)
    h, hm = radar_height(s), radar_median_height(s)
    moving = radar_moving(s, th.radar_threshold, th.active_feature)
    radar = decode_postures(model, [(h, hm, radar_offaxis(s), moving)])[0].astype(float)
    mocap = held_posture(rule.classify(s.mocap), max(int(round(rule.min_hold_s * FPS)), 1))
    return (s.t, radar, mocap, h, hm, s.mocap['head_y'], s.masks['window'],
            events[events.rec == rec])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs', action='store_true', help='copy the figures to docs/figures/')
    ap.add_argument('--example', default=EXAMPLE, help='recording of the timeline figure')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_category3', args, partial=False) as run_dir:
        events = pd.read_csv(latest('category3', 'events.csv'), keep_default_na=False, na_values=[''])
        mt = pd.read_csv(latest('category3', 'metrics.csv'))
        conf = pd.read_csv(latest('category3', 'posture_confusion.csv'), index_col=0)
        ps = pd.read_csv(latest('category3', 'posture_scores.csv'))
        es = pd.read_csv(latest('category3', 'event_scores.csv'))
        pk = pd.read_csv(latest('category3', 'per_kind.csv'))
        names = ('timeline', 'posture', 'posture_time', 'transitions', 'peak', 'per_participant')
        paths = {k: os.path.join(run_dir, f'c3_{k}.png') for k in names}
        print('timeline:', timeline_figure(*timeline_inputs(args.example, events), paths['timeline']))
        print('posture:', posture_figure(conf, ps, paths['posture']))
        print('posture time:', posture_time_figure(mt, paths['posture_time']))
        print('transitions:', transitions_figure(pk, events, paths['transitions']))
        print('peak:', peak_figure(events, mt, paths['peak']))
        table = per_participant_table(ps, es, mt, person_of)
        print('per participant (median, min, max):', per_participant_figure(table, paths['per_participant']))
        m = events[events.source == 'mocap']
        print(f'{events.rec.nunique()} recordings; MoCap transitions {len(m)}, matched {int(m.matched.sum())}, radar '
              f'{int((events.source == "radar").sum())}; pooled F1 '
              f'{2 * m.matched.sum() / (len(m) + (events.source == "radar").sum()):.3f}; '
              f'per recording F1 median {np.median(es.f1):.3f}')
        if args.docs:
            os.makedirs(DOCS_FIGURES, exist_ok=True)
            for p in paths.values():
                shutil.copy(p, DOCS_FIGURES)
            print(f'copied to {DOCS_FIGURES}')
