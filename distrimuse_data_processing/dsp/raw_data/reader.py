"""
Decode the raw ADC data recorded by a TI DCA1000 (.bin files), with the radar configuration of an mmWave Studio
.mmwave.json file.
"""

import json

import numpy as np

from .rf_config import RFConfig

N_RX = 4


def frame_shape(mmwave_json_file):
    """(loops, chirps, rx, samples) of one frame"""
    with open(mmwave_json_file, 'rt') as fp:
        devices = json.load(fp)['mmWaveDevices']
    assert len(devices) >= 1, "No mmWave Devices were configured"
    device = devices[0]
    assert not bool(device['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel']), \
        'IQ swap is turned on, turn it off!'
    assert device['rfConfig']["rlChanCfg_t"]["rxChannelEn"] == "0xF", "All 4 RX channels must be enabled"

    rf_config = RFConfig(device['rfConfig'])
    samples = rf_config.chirp_loop[0].profile_cfg.num_adc_samples
    assert all(chirp.profile_cfg.num_adc_samples == samples for chirp in rf_config.chirp_loop), \
        "All chirps must have the same number of adc samples"
    return rf_config.num_chirp_loops, rf_config.num_chirps, N_RX, samples


def frame_bytes(mmwave_json_file):
    """bytes per frame: complex int16 samples, 4 bytes each"""
    return int(np.prod(frame_shape(mmwave_json_file))) * 4


def read_adc_stream(mmwave_json_file, bin_files, file_offsets, frames_per_chunk=64):
    """
    Frames of a recording that the DCA1000 split into several .bin files. The files form one byte stream, cut
    every ~1 GB regardless of frame boundaries, so frames can straddle two files. Frame i is stream bytes
    [i * frame_bytes, (i + 1) * frame_bytes).

    :param bin_files: the recording's .bin files, in order
    :param file_offsets: stream byte offset of each file, increasing. Where a file doesn't start where the previous
        one ended (data missing), the incomplete frames on both sides of the gap are skipped.
    :return: generator of (frame index, frame), frame complex64 (loops, chirps, rx, samples); only complete frames
    """
    shape = frame_shape(mmwave_json_file)
    size = int(np.prod(shape)) * 4
    chunk_bytes = frames_per_chunk * size
    assert len(file_offsets) == len(bin_files), "one offset per file"

    pending = np.empty(0, np.uint8)  # bytes read but not decoded yet
    pending_start = 0                # stream offset of pending[0]
    next_frame = 0
    for bin_file, file_offset in zip(bin_files, file_offsets):
        if file_offset != pending_start + pending.size:
            assert file_offset > pending_start + pending.size, f"{bin_file} overlaps the previous file"
            pending = np.empty(0, np.uint8)  # an incomplete frame before a gap can't be completed
            pending_start = file_offset
        with open(bin_file, 'rb') as fp:
            while True:
                buf = np.fromfile(fp, dtype=np.uint8, count=chunk_bytes)
                if buf.size == 0:
                    break
                if pending.size:
                    buf = np.concatenate((pending, buf))
                skip = -pending_start % size  # bytes up to the next frame boundary (after a gap)
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
