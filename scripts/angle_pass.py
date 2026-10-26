"""
The angle pass: read raw recordings a second time and keep the per-channel complex spectrum of every cell that holds
something (mobility/radar/points.py), for angles and 3-D positions (mobility/radar/angle.py). Needs the cube and its
features (scripts/radar_pass.py, then any load_session, e.g. scripts/evaluate_category1.py): the noise floor comes
from them.

    python scripts/angle_pass.py --only person_1/scenario_1 person_10/scenario_2
    python scripts/angle_pass.py                    # all January recordings with a cube (~2 TB read: ~7 h)
    python scripts/angle_pass.py --device cpu --threads 8      # when the GPUs are busy: the network is the limit anyway

Recordings run one after the other: the network is the limit, parallel reads don't add throughput.
Writes OUTPUT_ROOT/<person>/<scenario>/points.npz and points_info.json (~27 GB for the campaign).
"""
import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.io.recordings import find_recordings, parse_recording  # noqa: E402
from mobility.radar.cube import cube_path  # noqa: E402
from mobility.radar.points import write_points  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', metavar='PERSON/SCENARIO')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
    ap.add_argument('--threads', type=int, default=None, help='CPU threads for the FFTs (with --device cpu)')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    recs = [parse_recording(r) for r in args.only] if args.only else find_recordings()
    no_cube = [r for r in recs if not os.path.exists(cube_path(*r))]
    if no_cube:
        print('skipped, no cube (run scripts/radar_pass.py first):', ' '.join('/'.join(r) for r in no_cube))
    failed = []
    for person, scenario in (r for r in recs if r not in no_cube):
        try:
            write_points(person, scenario, args.overwrite, torch.device(args.device) if args.device else None)
        except Exception:
            traceback.print_exc()
            failed.append(f'{person}/{scenario}')
    if failed:
        sys.exit(f'FAILED: {" ".join(failed)}')
