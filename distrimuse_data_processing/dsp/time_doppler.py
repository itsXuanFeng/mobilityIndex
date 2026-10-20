"""
Time-Doppler maps from the raw FMCW recordings (based on Reda El Hail's DSP code).

Each recording's .bin files are read as one byte stream and cut into frames. Per frame: the range-Doppler map of
TX0/RX0 in dB, and its maximum over range, one row of the time-Doppler map. Row j is frame j (j / 20 s after the
start of the capture); frames without data (missing from the files, or all-zero samples) are NaN rows.

Output per recording, in preprocessing/output/<campaign>/<person>/<scenario>/:
    --output time_doppler (default): time_doppler_max.npy, (frames, 128 Doppler bins), float64
    --output range_doppler:          range_doppler.npy, (frames, 128 range bins, 128 Doppler bins), --rd_dtype
Recordings that already have the output file are skipped, unless --overwrite.

Usage:
    python preprocessing/dsp/time_doppler.py --campaign August_2025 --workers 16
    python preprocessing/dsp/time_doppler.py --only person_1/scenario_1 --output range_doppler
"""
import argparse
import os
import sys
import time
import traceback
from collections import namedtuple
from multiprocessing import Pool

import numpy as np

from raw_data.reader import read_adc_stream, frame_bytes
from rangedoppler.rangedoppler import range_doppler_db, is_null, RANGE_BINS, DOPPLER_FFT
from recordings import RAW_ROOTS, find_scenarios, recording_files, frames_on_disk

HERE = os.path.dirname(os.path.abspath(__file__))
MMWAVE_CONFIG = os.path.join(HERE, 'rangedoppler', 'config.mmwave.json')  # same radar config for all recordings
OUTPUT_ROOT = os.path.normpath(os.path.join(HERE, '..', 'output'))
OUTPUT_FILES = {'time_doppler': 'time_doppler_max.npy', 'range_doppler': 'range_doppler.npy'}

Job = namedtuple('Job', 'campaign person scenario output rd_dtype overwrite')


def output_path(job):
    return os.path.join(OUTPUT_ROOT, job.campaign, job.person, job.scenario, OUTPUT_FILES[job.output])


def temp_path(path):
    """outputs are written here first and renamed when complete, so an interrupted run leaves no partial output"""
    return path[:-len('.npy')] + '.tmp.npy'


class FrameWriter:
    """writes an (n_frames, *shape) .npy file frame by frame, in increasing frame order; frames not written are NaN"""

    def __init__(self, f, n_frames, shape, dtype, block=64):
        self.f = f
        self.dtype = np.dtype(dtype)
        self.nan_block = np.full((block,) + shape, np.nan, self.dtype)
        self.next = 0
        np.lib.format.write_array_header_1_0(f, {'descr': np.lib.format.dtype_to_descr(self.dtype),
                                                 'fortran_order': False, 'shape': (n_frames,) + shape})

    def write(self, frame, data):
        self.skip_to(frame)
        self.f.write(data.astype(self.dtype).tobytes())
        self.next = frame + 1

    def skip_to(self, frame):
        while self.next < frame:
            k = min(len(self.nan_block), frame - self.next)
            self.f.write(self.nan_block[:k].tobytes())
            self.next += k


def time_doppler_map(paths, offsets, n_frames, rd_writer=None):
    """
    Time-Doppler map of a recording, (n_frames, DOPPLER_FFT), NaN for frames without data. With rd_writer, the
    range-Doppler map of every frame is written to it as well.
    :return: the map, and the number of null (all-zero) frames
    """
    td = np.full((n_frames, DOPPLER_FFT), np.nan)
    n_null = 0
    last = -1
    for frame, data in read_adc_stream(MMWAVE_CONFIG, paths, offsets):
        last = frame
        rd = range_doppler_db(data)
        doppler = rd.max(axis=0)
        if is_null(doppler):
            n_null += 1
            continue
        td[frame] = doppler
        if rd_writer is not None:
            rd_writer.write(frame, rd)
    assert last == n_frames - 1, 'frames read differ from the frames on disk'
    if rd_writer is not None:
        rd_writer.skip_to(n_frames)
    return td, n_null


def process_scenario(job):
    name = f'{job.person}/{job.scenario}'
    out = output_path(job)
    if os.path.exists(out) and not job.overwrite:
        print(f'{name}: already processed, skipped')
        return

    paths, offsets = recording_files(job.campaign, job.person, job.scenario)
    n_frames = frames_on_disk(paths, offsets, frame_bytes(MMWAVE_CONFIG))
    if n_frames == 0:
        raise ValueError(f'{name}: no complete frame in the files')
    print(f'{name}: {len(paths)} files, {n_frames} frames')

    start = time.time()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = temp_path(out)
    try:
        with open(tmp, 'wb') as f:
            if job.output == 'range_doppler':
                writer = FrameWriter(f, n_frames, (RANGE_BINS, DOPPLER_FFT), job.rd_dtype)
                td, n_null = time_doppler_map(paths, offsets, n_frames, writer)
            else:
                td, n_null = time_doppler_map(paths, offsets, n_frames)
                np.save(f, td)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    n_nan = int(np.isnan(td[:, 0]).sum())
    print(f'{name}: {OUTPUT_FILES[job.output]} done, {n_frames} frames, {n_nan} NaN '
          f'({n_nan - n_null} missing, {n_null} all-zero), {time.time() - start:.0f} s')


def process_safely(job):
    """process_scenario that prints and returns its error instead of raising it, so the other recordings go on"""
    try:
        process_scenario(job)
        return None
    except Exception:
        tb = traceback.format_exc()
        print(f'FAILED {job.person}/{job.scenario}:\n{tb}', flush=True)
        return tb.strip().splitlines()[-1]


def run_jobs(jobs, workers=1):
    """the error (or None) of every job"""
    try:
        if workers > 1:
            with Pool(workers) as pool:
                return list(pool.imap(process_safely, jobs))
        return [process_safely(job) for job in jobs]
    finally:
        # on Ctrl-C the pool kills its workers (SIGTERM) before they can remove their temporary files
        for job in jobs:
            tmp = temp_path(output_path(job))
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign', default='January_2026', choices=sorted(RAW_ROOTS))
    ap.add_argument('--only', nargs='+', default=None, metavar='PERSON/SCENARIO',
                    help='process only these recordings, e.g. person_1/scenario_1')
    ap.add_argument('--workers', type=int, default=1, help='recordings processed in parallel')
    ap.add_argument('--overwrite', action='store_true', help='recompute recordings that already have an output file')
    ap.add_argument('--output', default='time_doppler', choices=sorted(OUTPUT_FILES),
                    help='time_doppler (default) or range_doppler (large: ~0.8 GB per 10 min of recording in float32)')
    ap.add_argument('--rd_dtype', default='float32', choices=['float16', 'float32', 'float64'],
                    help='dtype of range_doppler.npy')
    args = ap.parse_args()

    scenarios = find_scenarios(args.campaign)
    if args.only:
        wanted = {tuple(s.strip('/').split('/')) for s in args.only}
        missing = wanted - set(scenarios)
        assert not missing, f'not found: {sorted(missing)}'
        scenarios = [s for s in scenarios if s in wanted]
    jobs = [Job(args.campaign, person, scenario, args.output, args.rd_dtype, args.overwrite)
            for person, scenario in scenarios]
    print(f'{len(jobs)} recordings -> {os.path.join(OUTPUT_ROOT, args.campaign)}')

    errors = run_jobs(jobs, args.workers)
    failed = [(job, error) for job, error in zip(jobs, errors) if error]
    if failed:
        print(f'{len(failed)} of {len(jobs)} recordings FAILED (no output written for them):'
              + ''.join(f'\n  {job.person}/{job.scenario}: {error}' for job, error in failed))
        sys.exit(1)
