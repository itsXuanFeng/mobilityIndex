"""
The one expensive step: read a raw recording once and store its per-frame range-Doppler power.

Output in OUTPUT_ROOT/<person>/<scenario>/:
    rd12_db.npy   (T, 128 range, 128 Doppler) float16, dB of the mean power over the 12 virtual channels
    pass_timing.txt
Row j is frame j (j / 20 s after radar frame 0); frames without data are NaN.
Everything downstream (features, Categories 1-3) is computed from these cubes, so the 13 GB of raw data per recording
is read only once. Reading is limited by the network (~110 MB/s on a 1 Gbit/s link), not by the GPU.
"""
import os
import queue
import threading
import time

import numpy as np
import torch

from ..config import N_DOPPLER, N_RANGE, OUTPUT_ROOT
from ..io.raw import frame_bytes, frames_on_disk, read_frames, recording_files
from .dsp import RangeDoppler

BATCH = 64


def recording_dir(person, scenario):
    return os.path.join(OUTPUT_ROOT, person, scenario)


def cube_path(person, scenario):
    return os.path.join(recording_dir(person, scenario), 'rd12_db.npy')


def load_cube(person, scenario):
    """(T, 128, 128) float16 dB, memory-mapped"""
    return np.load(cube_path(person, scenario), mmap_mode='r')


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
    out = recording_dir(person, scenario)
    path = cube_path(person, scenario)
    if not overwrite and os.path.exists(path):
        print(f'{person}/{scenario}: cube exists, skipped')
        return
    os.makedirs(out, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = RangeDoppler(device)
    paths, offsets = recording_files(person, scenario)
    T = frames_on_disk(paths, offsets, frame_bytes())
    tmp = path[:-4] + '.tmp.npy'
    cube = np.lib.format.open_memmap(tmp, 'w+', np.float16, (T, N_RANGE, N_DOPPLER))
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
            p = rd.power(x)                                          # (n, tx, rx, range, Doppler)
            null = (p.flatten(1).amax(1) == 0).cpu().numpy()         # all-zero frames: no data
            db = (10 * torch.log10(p.mean(dim=(1, 2)) + 1e-30)).cpu().numpy()   # mean over the 12 channels
            db[null] = np.nan
            cube[i0:i0 + len(x)] = db
            t_compute += time.time() - t
        cube.flush()
        del cube
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    total = time.time() - t_start
    msg = (f'{person}/{scenario}: {T} frames, {total:.0f} s = {1e3 * total / T:.1f} ms/frame; waiting for data '
           f'{t_wait:.0f} s, GPU {t_compute:.0f} s ({1e3 * t_compute / T:.2f} ms/frame)')
    with open(os.path.join(out, 'pass_timing.txt'), 'w') as fp:
        fp.write(msg + '\n')
    print(msg, flush=True)
