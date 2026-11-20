"""
Timeline of one recording after alignment: labels, MoCap speed, radar features, range-time and Doppler-time maps.

    python scripts/plot_timeline.py person_1/scenario_1

Writes <person>_<scenario>_timeline.png to its run folder
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.experiment import add_run_args, run  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.diagnostic import timeline  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', metavar='PERSON/SCENARIO')
    add_run_args(ap)
    args = ap.parse_args()
    with run('plot_timeline', args, partial=False) as run_dir:
        person, scenario = parse_recording(args.recording)
        path = os.path.join(run_dir, f'{person}_{scenario}_timeline.png')
        timeline(load_session(person, scenario), path)
        print(path)
