"""Which radar recordings exist, when they started, and their activity labels."""
import glob
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import FMCW_ROOT, LABEL_ROOT

CET = timezone(timedelta(hours=1))   # the capture PC logs local time; January has no DST


def natural_key(s):
    """sort key that orders numbers numerically: 'person_2' < 'person_10'"""
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def find_recordings():
    """(person, scenario) of every radar recording with .bin files"""
    files = glob.glob(f'{FMCW_ROOT}/person_*/scenario_*/*_Raw_*.bin')
    found = {tuple(f.split('/')[-3:-1]) for f in files}
    return sorted(found, key=lambda ps: natural_key('/'.join(ps)))


def parse_recording(s):
    """'person_1/scenario_1' -> ('person_1', 'scenario_1')"""
    person, scenario = s.strip('/').split('/')
    return person, scenario


def capture_times(person, scenario):
    """capture start and end (epoch s, UTC) from the DCA1000 log. Logged to the whole second only: radar frame 0 is
    some fraction of a second after the start; mobility.align finds the exact offset."""
    with open(f'{FMCW_ROOT}/{person}/{scenario}/test2x_1_Raw_LogFile.csv') as fp:
        txt = fp.read()

    def parse(key):
        stamp = re.search(key + r' - (.+)', txt).group(1).strip()
        return datetime.strptime(stamp, '%a %b %d %H:%M:%S %Y').replace(tzinfo=CET).timestamp()

    return parse('Capture start time'), parse('Capture end time')


def labels(person, scenario):
    """scripted activities: label, start, end (epoch s, UTC), as recorded by the operator"""
    df = pd.read_csv(f'{LABEL_ROOT}/{person}/{scenario}/timestamps.csv')
    df.columns = ['label', 'start', 'end']
    df['label'] = df.label.str.strip().str.rstrip('.')
    return df
