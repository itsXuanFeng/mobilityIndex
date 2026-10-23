"""
Timeline of one recording after alignment: labels, MoCap speed, radar features, range-time and Doppler-time maps.

    python scripts/plot_timeline.py person_1/scenario_1

Writes OUTPUT_ROOT/plots/<person>_<scenario>_timeline.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import OUTPUT_ROOT  # noqa: E402
from mobility.io.recordings import parse_recording  # noqa: E402
from mobility.plots.diagnostic import timeline  # noqa: E402
from mobility.session import load_session  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', metavar='PERSON/SCENARIO')
    args = ap.parse_args()
    person, scenario = parse_recording(args.recording)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'plots'), exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, 'plots', f'{person}_{scenario}_timeline.png')
    timeline(load_session(person, scenario), path)
    print(path)
