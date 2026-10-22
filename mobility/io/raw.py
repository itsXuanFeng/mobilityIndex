"""
Raw DCA1000 recordings: locating the .bin files of a recording and decoding them into frames.
Byte layout and missing-data cases: docs/raw_data_january.md. Adapted from the original preprocessing code
(commit dc5c08c, distrimuse_data_processing/dsp: reader.py, recordings.py).
"""
import json
import os
import re

import numpy as np

from ..config import FMCW_ROOT

MMWAVE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.mmwave.json')
N_RX = 4

# The DCA1000 splits a recording into .bin files of FILE_BYTES each (1024 MB rounded down to whole 1456-byte packets).
# Raw_<k> starts at byte k * FILE_BYTES of the recording, also when earlier files are truncated or missing.
FILE_BYTES = 1073741760

# Raw_15 doesn't align with its position in the recording (Raw_12-14 are missing and Raw_11 is truncated)
EXCLUDE = {('person_6', 'scenario_3'): ['test2x_1_Raw_15.bin']}


def frame_shape(mmwave_json=MMWAVE_CONFIG):
    """(loops, chirps, rx, samples) of one frame"""
    with open(mmwave_json) as fp:
        rf = json.load(fp)['mmWaveDevices'][0]
    assert not bool(rf['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel']), 'IQ swap must be off'
    rf = rf['rfConfig']
    assert rf['rlChanCfg_t']['rxChannelEn'] == '0xF', 'all 4 RX channels must be enabled'
    frame = rf['rlFrameCfg_t']
    samples = {p['rlProfileCfg_t']['numAdcSamples'] for p in rf['rlProfiles']}
    assert len(samples) == 1, 'all chirps must have the same number of ADC samples'
    return frame['numLoops'], frame['chirpEndIdx'] - frame['chirpStartIdx'] + 1, N_RX, samples.pop()


def frame_bytes():
    """bytes per frame: complex int16 samples, 4 bytes each"""
    return int(np.prod(frame_shape())) * 4


def recording_files(person, scenario):
    """the recording's .bin files in order, and the byte offset of each in the recording"""
    folder = os.path.join(FMCW_ROOT, person, scenario)
    excluded = EXCLUDE.get((person, scenario), [])
    numbered = {}
    for name in os.listdir(folder):
        match = re.search(r'_Raw_(\d+)\.bin$', name)
        if match and name not in excluded:
            numbered[int(match.group(1))] = os.path.join(folder, name)
    paths = [numbered[k] for k in sorted(numbered)]
    for path in paths:
        assert os.path.getsize(path) <= FILE_BYTES, f'{path} is larger than {FILE_BYTES} bytes'
    return paths, [k * FILE_BYTES for k in sorted(numbered)]


def frames_on_disk(paths, offsets, size):
    """number of frames up to the last complete frame on disk (the last frame whose bytes lie in one contiguous run
    of files)"""
    spans = []
    for path, offset in zip(paths, offsets):
        end = offset + os.path.getsize(path)
        if spans and spans[-1][1] == offset:
            spans[-1][1] = end
        else:
            spans.append([offset, end])
    for start, end in reversed(spans):
        if end // size > -(-start // size):
            return end // size
    return 0


def read_frames(paths, offsets, frames_per_chunk=64):
    """
    Frames of a recording. The files form one byte stream cut every ~1 GB regardless of frame boundaries, so frames
    can straddle two files; frame i is stream bytes [i * size, (i + 1) * size). Frames next to missing data are skipped.
    :return: generator of (frame index, frame complex64 (loops, chirps, rx, samples))
    """
    shape = frame_shape()
    size = int(np.prod(shape)) * 4
    pending = np.empty(0, np.uint8)   # bytes read but not decoded yet
    pending_start = 0                 # stream offset of pending[0]
    next_frame = 0
    for bin_file, file_offset in zip(paths, offsets):
        if file_offset != pending_start + pending.size:
            assert file_offset > pending_start + pending.size, f'{bin_file} overlaps the previous file'
            pending = np.empty(0, np.uint8)
            pending_start = file_offset
        with open(bin_file, 'rb') as fp:
            while True:
                buf = np.fromfile(fp, dtype=np.uint8, count=frames_per_chunk * size)
                if buf.size == 0:
                    break
                if pending.size:
                    buf = np.concatenate((pending, buf))
                skip = -pending_start % size   # bytes up to the next frame boundary (after a gap)
                n = max(0, (buf.size - skip) // size)
                used = skip + n * size if n else 0
                pending = buf[used:].copy()
                if n == 0:
                    continue
                first = (pending_start + skip) // size
                pending_start += used
                if first != next_frame:
                    print(f'{bin_file}: gap in the data, frames {next_frame}-{first - 1} missing or incomplete')
                next_frame = first + n
                # every 8 bytes are <hhhh = i0, i1, q0, q1: samples i0 + j*q0 and i1 + j*q1
                q = buf[skip:skip + n * size].view('<i2').reshape((n,) + shape[:-1] + (shape[-1] // 2, 4))
                frames = np.empty((n,) + shape, np.csingle)
                frames.real[..., 0::2] = q[..., 0]
                frames.real[..., 1::2] = q[..., 1]
                frames.imag[..., 0::2] = q[..., 2]
                frames.imag[..., 1::2] = q[..., 3]
                for k in range(n):
                    yield first + k, frames[k]
