"""
The one expensive step: read a raw recording once and store its per-frame range-Doppler power.

Output in OUTPUT_ROOT/<person>/<scenario>/:
    rd12_db.npy      (T, 128 range, 128 Doppler) float16, dB of the mean power over the 12 virtual channels
    pass_info.json   frames without data, frames damaged by lost packets, timing
Row j is frame j (j / 20 s after radar frame 0). NaN rows: frames missing from the files, all-zero frames, and frames
damaged by packets the DCA1000 lost and filled with zeros (a gap in the samples spreads energy over all Doppler bins,
which would look like movement).
Everything downstream (features, Categories 1-3) is computed from these cubes, so the 13 GB of raw data per recording
is read only once. Reading is limited by the network (~110 MB/s on a 1 Gbit/s link), not by the GPU.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import N_DOPPLER, N_RANGE, recording_dir
from ..io.raw import frame_bytes, frames_on_disk, read_frames, recording_files
from ..io.recordings import lost_packets
from .dsp import RangeDoppler

BATCH = 64
# A lost 1456-byte packet zero-fills 364 consecutive complex samples; real ADC data practically never has I = Q = 0
# for this many samples in a row. Shorter zero runs (a lost packet cut by a frame boundary) barely distort the frame.
MIN_ZERO_RUN = 16


def cube_path(person, scenario):
    return os.path.join(recording_dir(person, scenario), 'rd12_db.npy')


def info_path(person, scenario):
    return os.path.join(recording_dir(person, scenario), 'pass_info.json')


def load_cube(person, scenario):
    """(T, 128, 128) float16 dB, memory-mapped"""
    return np.load(cube_path(person, scenario), mmap_mode='r')


def zero_filled(frames, min_run=MIN_ZERO_RUN):
    """(n,) bool: frames with at least min_run consecutive samples (stream order) exactly 0, i.e. a lost packet"""
    z = (frames == 0).reshape(len(frames), 1, -1).float()
    return F.avg_pool1d(z, min_run, stride=1).amax(-1).squeeze(1) > 1 - 1e-6


def _batches(paths, offsets, q):
    """reader thread: batches of consecutive frames, so reading overlaps with the GPU work"""
    buf, idx = [], []
    try:
        for i, frame in read_frames(paths, offsets, frames_per_chunk=BATCH):
            if buf and (i != idx[-1] + 1 or len(buf) == BATCH):
                q.put((idx[0], np.stack(buf)))
                buf, idx = [], []
            buf.append(frame)
            idx.append(i)
        if buf:
            q.put((idx[0], np.stack(buf)))
    finally:
        q.put(None)


def write_cube(person, scenario, overwrite=False, device=None):
    """the cube of one recording; skipped when it and its pass_info.json exist (unless overwrite)"""
    out = recording_dir(person, scenario)
    path = cube_path(person, scenario)
    if not overwrite and os.path.exists(path) and os.path.exists(info_path(person, scenario)):
        print(f'{person}/{scenario}: cube exists, skipped', flush=True)
        return
    os.makedirs(out, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = RangeDoppler(device)
    paths, offsets = recording_files(person, scenario)
    T = frames_on_disk(paths, offsets, frame_bytes())
    tmp = path[:-4] + '.tmp.npy'
    cube = np.lib.format.open_memmap(tmp, 'w+', np.float16, (T, N_RANGE, N_DOPPLER))
    n_read, null_frames, damaged_frames = 0, [], []
    try:
        cube[:] = np.nan
        q = queue.Queue(maxsize=8)
        threading.Thread(target=_batches, args=(paths, offsets, q), daemon=True).start()
        t_wait = t_compute = 0.0
        t_start = time.time()
        while True:
            t = time.time()
            item = q.get()
            t_wait += time.time() - t
            if item is None:
                break
            t = time.time()
            i0, x = item
            x = torch.as_tensor(x, device=device)
            p = rd.power(x)                                          # (n, tx, rx, range, Doppler)
            null = (p.flatten(1).amax(1) == 0).cpu().numpy()         # all-zero frames: no data
            damaged = zero_filled(x).cpu().numpy() & ~null
            db = (10 * torch.log10(p.mean(dim=(1, 2)) + 1e-30)).cpu().numpy()   # mean over the 12 channels
            db[null | damaged] = np.nan
            cube[i0:i0 + len(db)] = db
            n_read += len(db)
            null_frames += (i0 + np.flatnonzero(null)).tolist()
            damaged_frames += (i0 + np.flatnonzero(damaged)).tolist()
            t_compute += time.time() - t
        cube.flush()
        del cube
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    total = time.time() - t_start
    info = dict(frames=T, missing_frames=T - n_read, null_frames=null_frames, damaged_frames=damaged_frames,
                lost_packets_logged=lost_packets(person, scenario), seconds=round(total, 1),
                waiting_for_data_s=round(t_wait, 1), gpu_s=round(t_compute, 1))
    with open(info_path(person, scenario), 'w') as fp:
        json.dump(info, fp)
    old = os.path.join(out, 'pass_timing.txt')
    if os.path.exists(old):
        os.remove(old)
    print(f'{person}/{scenario}: {T} frames, {T - n_read} missing, {len(null_frames)} all-zero, '
          f'{len(damaged_frames)} damaged by lost packets ({info["lost_packets_logged"]} logged); {total:.0f} s = '
          f'{1e3 * total / T:.1f} ms/frame, waiting for data {t_wait:.0f} s', flush=True)
