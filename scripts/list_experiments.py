"""
The experiment runs in output/experiments/, oldest first: start time, status, git commit, whether it is the latest,
and the runs it read. Not an experiment itself (writes nothing).

    python scripts/list_experiments.py                  # every run
    python scripts/list_experiments.py category2 bout_rule   # these experiments only
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobility.config import EXPERIMENTS_DIR, LATEST_DIR  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='experiment names (default: all)')
    args = ap.parse_args()
    latest = {os.path.realpath(os.path.join(LATEST_DIR, n)) for n in os.listdir(LATEST_DIR)} \
        if os.path.isdir(LATEST_DIR) else set()
    rows = []
    for d in sorted(os.listdir(EXPERIMENTS_DIR)) if os.path.isdir(EXPERIMENTS_DIR) else []:
        path = os.path.join(EXPERIMENTS_DIR, d)
        try:
            with open(os.path.join(path, 'run.json')) as f:
                info = json.load(f)
        except (OSError, ValueError):
            info = dict(name=d.rsplit('_', 2)[0], status='?')
        if args.names and info.get('name') not in args.names:
            continue
        commit = (info.get('git') or {}).get('commit') or ''
        dirty = '+' if (info.get('git') or {}).get('dirty') else ''
        rows.append((info.get('started') or '', d, info.get('status', '?'), commit[:7] + dirty,
                     '*' if os.path.realpath(path) in latest else '',
                     ', '.join(f'{k}={v}' for k, v in (info.get('inputs') or {}).items())))
    rows.sort()
    if not rows:
        sys.exit('no runs')
    w = max(len(r[1]) for r in rows)
    print(f'{"started":19s}  {"run":{w}s}  {"status":9s}  {"commit":8s}  L  inputs')
    for started, d, status, commit, lat, inputs in rows:
        print(f'{started:19s}  {d:{w}s}  {status:9s}  {commit:8s}  {lat:1s}  {inputs}')
    print('\nL: * = output/latest/<name> points at this run; commit + = the tree was dirty (diff.patch in the run)')
