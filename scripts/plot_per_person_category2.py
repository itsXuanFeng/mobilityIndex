"""
Category 2 per person: signed error in % of the MoCap value, 100 x (radar - MoCap) / MoCap per recording, of the
number of bouts, the mean bout duration, the SD of the bout duration and the fragmentation index. Bars: the person's
mean over their recordings (red: radar too high, blue: too low); dots: one recording each. All on one side = a
systematic error for that person. One y-scale for all panels, so the metrics can be compared.
Reads OUTPUT_ROOT/category2/metrics.csv and bouts.csv: run scripts/evaluate_category2.py first.

    python scripts/plot_per_person_category2.py

Writes OUTPUT_ROOT/plots/per_person/per_person_bouts_bias.png
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import PLOTS_DIR, category_dir  # noqa: E402
from mobility.io.recordings import natural_key  # noqa: E402
from mobility.plots.per_person import BOUT_PANELS, bouts_bias_figure  # noqa: E402
from mobility.utils import person_of  # noqa: E402

IN = category_dir(2)
OUT = os.path.join(PLOTS_DIR, 'per_person')


def signed_errors():
    """per recording and metric: person, radar - MoCap in % of MoCap. The SD of the bout duration comes from the
    bouts"""
    mt = pd.read_csv(os.path.join(IN, 'metrics.csv'))
    b = pd.read_csv(os.path.join(IN, 'bouts.csv'))
    sd = b.groupby(['rec', 'source']).duration_s.std(ddof=1).unstack('source').reset_index()
    sd = sd.assign(metric='bout_duration_sd')
    mt = pd.concat([mt[['rec', 'metric', 'radar', 'mocap']], sd[['rec', 'metric', 'radar', 'mocap']]])
    return mt.assign(person=person_of(mt.rec), err=100 * (mt.radar - mt.mocap) / mt.mocap)


if __name__ == '__main__':
    e = signed_errors()
    persons = sorted(e.person.unique(), key=natural_key)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'per_person_bouts_bias.png')
    bouts_bias_figure(e, persons, path)
    print(f'wrote {path}')
    t = e.groupby(['metric', 'person']).err.mean().unstack('metric')[[k for k, _ in BOUT_PANELS]]
    print(t.reindex(persons).round(1).to_string())
