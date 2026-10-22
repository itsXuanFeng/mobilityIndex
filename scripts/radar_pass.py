"""
Read raw recordings once and write their range-Doppler cubes (mobility/radar/cube.py).

    python scripts/radar_pass.py --only person_1/scenario_1 person_5/scenario_3
    python scripts/radar_pass.py                     # all January recordings (~2 TB: >= 5 h on a 1 Gbit/s link)

Recordings run one after the other: the network is the limit, parallel reads don't add throughput.
Pick a free GPU with CUDA_VISIBLE_DEVICES.
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.io.recordings import find_recordings, parse_recording  # noqa: E402
from mobility.radar.cube import write_cube  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    recs = [parse_recording(r) for r in args.only] if args.only else find_recordings()
    failed = []
    for person, scenario in recs:
        try:
            write_cube(person, scenario, args.overwrite)
        except Exception:
            traceback.print_exc()
            failed.append(f'{person}/{scenario}')
    if failed:
        sys.exit(f'FAILED: {" ".join(failed)}')
