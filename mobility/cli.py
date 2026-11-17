"""Helpers shared by the command-line scripts in scripts/: which recordings, running them in parallel, printing."""
import traceback
from multiprocessing import Pool

import pandas as pd

from .io.recordings import find_recordings, parse_recording


def recordings(only=None):
    """(person, scenario) of the recordings given as 'person_x/scenario_y' (--only), else of every radar recording"""
    return [parse_recording(r) for r in only] if only else find_recordings()


def parallel_map(fn, jobs, workers):
    """[fn(job) for job in jobs], in a pool of `workers` processes when more than one. Pool workers are forked, so
    module-level data the script filled before the call is visible to fn."""
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


def error_summary():
    """the last line of the exception being handled, e.g. "KeyError: 'ke'" """
    return traceback.format_exc().strip().splitlines()[-1]


def wide_tables(max_columns=None):
    """pandas prints wide tables on one line"""
    pd.set_option('display.width', 200)
    if max_columns:
        pd.set_option('display.max_columns', max_columns)
