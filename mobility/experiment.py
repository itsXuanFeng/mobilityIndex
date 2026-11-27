"""Experiment runs: every script writes into its own timestamped folder, and reads the results of other experiments
from their latest finished run.

    output/experiments/<name>[-<tag>]_<YYYY-MM-DD_HHMMSS>/   one run: its tables and figures, plus
        log.txt      everything the script printed (stdout and stderr, also from worker processes)
        run.json     name, tag, command, start / end, status, git commit (and whether the tree was dirty),
                     arguments, and the runs of other experiments it read (inputs)
        diff.patch   `git diff HEAD` when the tree was dirty, so the run can be reproduced
    output/latest/<name> -> ../experiments/<name>_<...>       the last finished, complete, untagged run of <name>

In a script:

    ap = argparse.ArgumentParser(...)
    add_run_args(ap)                       # --tag, --use NAME=RUN, --promote
    args = ap.parse_args()
    with run('category2', args, partial=bool(args.only)) as out:
        th = pd.read_csv(latest('category1', 'thresholds.csv'))   # recorded as an input of this run
        table.to_csv(f'{out}/metrics.csv')

A run updates output/latest/<name> only when it finishes without error, has no --tag and is not partial (a subset of
the recordings), unless --promote is given; downstream experiments therefore never pick up an exploration or a quick
check by accident. --use category1=category1_2026-11-02_143000 makes this run read that run instead of the latest.
"""
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from contextlib import contextmanager

from .config import EXPERIMENTS_DIR, LATEST_DIR, REPO

# the script that produces each experiment (for error messages and the README)
SCRIPTS = {
    'radar_pass': 'scripts/radar_pass.py',
    'angle_pass': 'scripts/angle_pass.py',
    'mocap_threshold': 'scripts/calibrate_mocap_threshold.py',
    'bout_rule': 'scripts/calibrate_bout_rule.py',
    'posture_rule': 'scripts/calibrate_posture.py',
    'category1': 'scripts/evaluate_category1.py',
    'category2': 'scripts/evaluate_category2.py',
    'category3': 'scripts/evaluate_category3.py',
    'personal': 'scripts/evaluate_personal_calibration.py',
    'mounting_check': 'scripts/check_mounting_height.py',
    'floor_sensitivity': 'scripts/sensitivity_floor.py',
    'so_cfar': 'scripts/prototype_so_cfar.py',
    'radar_orientation': 'scripts/fit_radar_orientation.py',
    'radar_position': 'scripts/fit_radar_position.py',
    'plot_agreement': 'scripts/plot_agreement.py',
    'plot_bouts': 'scripts/plot_bouts.py',
    'plot_transitions': 'scripts/plot_transitions.py',
    'plot_category1': 'scripts/plot_category1.py',
    'plot_category2': 'scripts/plot_category2.py',
    'plot_category3': 'scripts/plot_category3.py',
    'plot_per_person': 'scripts/plot_per_person.py',
    'plot_per_person_category2': 'scripts/plot_per_person_category2.py',
    'plot_per_person_category3': 'scripts/plot_per_person_category3.py',
    'plot_per_person_personal': 'scripts/plot_per_person_personal.py',
    'plot_timeline': 'scripts/plot_timeline.py',
}

TIME_FORMAT = '%Y-%m-%d_%H%M%S'

_current = None     # folder of the run of this process (forked workers inherit it)
_used = {}          # experiment name -> run folder this process reads (pinned at first use, or set by --use)


def add_run_args(ap):
    """the arguments every experiment script takes"""
    g = ap.add_argument_group('experiment run')
    g.add_argument('--tag', help='variant name, added to the folder name (a tagged run does not become latest)')
    g.add_argument('--use', nargs='+', default=[], metavar='NAME=RUN',
                   help='read this run of experiment NAME instead of its latest, '
                        'e.g. category1=category1_2026-11-02_143000')
    g.add_argument('--promote', action='store_true', help='make this run the latest even if tagged or partial')
    return ap


def _resolve_run(run_name):
    """a run given by folder name or path -> its absolute folder"""
    path = run_name if os.path.isabs(run_name) else os.path.join(EXPERIMENTS_DIR, run_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'{path}: no such run')
    return os.path.realpath(path)


def latest(name, *parts):
    """path inside the run of experiment `name` that this process reads: the one given with --use, else
    output/latest/<name> (resolved once per process, so a run finishing meanwhile doesn't mix two runs). Recorded as
    an input of the current run."""
    if name not in _used:
        link = os.path.join(LATEST_DIR, name)
        if not os.path.exists(link):
            script = SCRIPTS.get(name, f'the script of {name}')
            raise FileNotFoundError(f'{link} missing: no finished run of {name!r}; run {script} first')
        _used[name] = os.path.realpath(link)
    _record_input(name, _used[name])
    path = os.path.join(_used[name], *parts)
    if parts and not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: the {name} run {os.path.basename(_used[name])} did not write it')
    return path


def _record_input(name, folder):
    """append to <run>/.inputs (O_APPEND: safe from forked workers); collected into run.json at the end. Read before
    the run started: recorded when it starts."""
    if _current is None:
        _record_input.pending[name] = folder
        return
    key = (os.getpid(), name)
    if key in _record_input.seen:
        return
    _record_input.seen.add(key)
    with open(os.path.join(_current, '.inputs'), 'a') as f:
        f.write(f'{name}\t{folder}\n')


_record_input.seen = set()
_record_input.pending = {}


def _git(*cmd):
    try:
        return subprocess.run(['git', '-C', REPO, *cmd], capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001 - git missing or not a repository: the run goes on without it
        return ''


def _git_info(folder):
    commit = _git('rev-parse', 'HEAD').strip()
    status = _git('status', '--porcelain', '--untracked-files=normal').splitlines()
    dirty = [s for s in status if not s.startswith('??')]
    untracked = [s[3:] for s in status if s.startswith('??')]
    # untracked code (a new prototype script, say) goes into the patch too, as new files
    new_code = [f for f in untracked if f.endswith('.py') and os.path.getsize(os.path.join(REPO, f)) < 1_000_000]
    if dirty or new_code:
        with open(os.path.join(folder, 'diff.patch'), 'w') as f:
            f.write(_git('diff', 'HEAD'))
            for p in new_code:
                f.write(_git('diff', '--no-index', '--', '/dev/null', p))
    return dict(commit=commit or None, dirty=bool(dirty), changed=[s[3:] for s in dirty], untracked=untracked)


class _Tee:
    """writes to the terminal and to the run's log"""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        if '\n' in s:           # pool workers exit without flushing: lines would be lost on a piped stdout
            self.stream.flush()
        try:
            self.log.write(s)
        except ValueError:   # log closed (a worker outliving the run)
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.log.flush()
        except ValueError:
            pass

    def __getattr__(self, k):
        return getattr(self.stream, k)


def _new_folder(name, tag):
    stem = f'{name}-{tag}' if tag else name
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    now = datetime.datetime.now()
    base = os.path.join(EXPERIMENTS_DIR, f'{stem}_{now.strftime(TIME_FORMAT)}')
    folder, i = base, 2
    while True:
        try:
            os.makedirs(folder)
            return folder, now
        except FileExistsError:
            folder, i = f'{base}-{i}', i + 1


def _write_json(folder, info):
    tmp = os.path.join(folder, '.run.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    os.replace(tmp, os.path.join(folder, 'run.json'))


def promote(name, folder):
    """point output/latest/<name> at `folder` (atomically)"""
    os.makedirs(LATEST_DIR, exist_ok=True)
    link = os.path.join(LATEST_DIR, name)
    tmp = f'{link}.tmp{os.getpid()}'
    os.symlink(os.path.relpath(folder, LATEST_DIR), tmp)
    os.replace(tmp, link)


@contextmanager
def run(name, args=None, partial=False):
    """create the run folder of experiment `name`, log everything printed to it, record the run in run.json; yields
    the folder. `args`: the parsed arguments (recorded; --tag, --use, --promote read from them). `partial`: the run
    covers only part of the data (--only, --persons): it doesn't become latest unless --promote."""
    global _current
    if name not in SCRIPTS:
        raise ValueError(f'unknown experiment {name!r}; add it to mobility.experiment.SCRIPTS')
    tag = getattr(args, 'tag', None)
    for u in getattr(args, 'use', None) or []:
        k, _, v = u.partition('=')
        if not v:
            raise ValueError(f'--use {u}: expected NAME=RUN')
        _used[k] = _resolve_run(v)
    folder, start = _new_folder(name, tag)
    info = dict(name=name, tag=tag, folder=os.path.basename(folder), status='running', partial=bool(partial),
                started=start.isoformat(timespec='seconds'), finished=None,
                command=' '.join([os.path.relpath(sys.argv[0], REPO)] + sys.argv[1:]), cwd=os.getcwd(),
                host=socket.gethostname(), python=platform.python_version(), executable=sys.executable,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                args={k: v for k, v in vars(args).items()} if args is not None else None,
                git=_git_info(folder), inputs={})
    _write_json(folder, info)

    log = open(os.path.join(folder, 'log.txt'), 'a', buffering=1)
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
    _current = folder
    for k, v in _record_input.pending.items():
        _record_input(k, v)
    _record_input.pending.clear()
    print(f'[run] {folder}')
    try:
        yield folder
        info['status'] = 'done'
    except KeyboardInterrupt:
        info['status'] = 'interrupted'
        raise
    except SystemExit as e:     # sys.exit() / sys.exit(0): done; sys.exit('message') or a code: failed
        info['status'] = 'done' if e.code in (None, 0) else 'failed'
        if e.code not in (None, 0):
            log.write(f'exit: {e.code}\n')
        raise
    except BaseException:
        info['status'] = 'failed'
        log.write(traceback.format_exc())
        raise
    finally:
        info['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        inputs_file = os.path.join(folder, '.inputs')
        if os.path.exists(inputs_file):
            with open(inputs_file) as f:
                for line in f:
                    k, v = line.rstrip('\n').split('\t')
                    info['inputs'][k] = os.path.relpath(v, EXPERIMENTS_DIR)
            os.remove(inputs_file)
        promoted = info['status'] == 'done' and (getattr(args, 'promote', False) or not (tag or partial))
        if promoted:
            promote(name, folder)
        info['latest'] = promoted
        _write_json(folder, info)
        print(f'[run] {info["status"]}: {folder}' + (f' (now output/latest/{name})' if promoted else ''))
        sys.stdout, sys.stderr = out, err
        _current = None
        log.close()
