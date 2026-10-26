"""
The angle pass (product 3 in SCOPE.md): a second read of a raw recording that keeps, for every range-Doppler cell that
holds something, its complex value on each of the 12 virtual channels. The phases across the channels carry the
direction; mobility.radar.angle turns them into angles and positions.

Only the measurement is stored: no angle method, no RX sign correction, no TDM Doppler compensation, no radar
orientation. Those are cheap and can be redone from these files without reading the raw data again (13 GB per
recording, network-bound: ~7 h for the campaign).

Output in OUTPUT_ROOT/<person>/<scenario>/:
    points.npz          one row per kept cell ("point"), sorted by frame:
        frame   (N,) int32           frame index = row of the cube (rd12_db.npy)
        rbin    (N,) uint8           range bin
        dbin    (N,) uint8           Doppler bin, fftshifted as in the cube (N_DOPPLER // 2 = zero velocity)
        db      (N,) float16         12-channel mean power (dB), as in the cube
        iq      (N, 12, 2) float16   real, imag per virtual channel (TX outer, RX inner: chirp k = TX k), divided by
                                     the largest magnitude of the 12, so float16 keeps the phases and relative amplitudes
    points_info.json    selection, counts, frames skipped, timing

Kept cells: range >= MIN_RANGE, any Doppler bin but zero velocity, more than POINT_DB above the cell's noise floor.
The floor is the recording's own (mobility.session.noise_floor, from the cube), so the cube and its features must exist
first. Slow bins are kept on purpose: breathing and small movements (~0.04 m/s) may locate a person who keeps still.
Frames without data or damaged by lost packets (NaN in the cube) are skipped, as in the cube.
"""
import json
import os
import queue
import threading
import time

import numpy as np
import torch

from ..config import ABOVE_FLOOR_DB, MIN_RANGE, N_DOPPLER, RANGE_AXIS, recording_dir
from ..io.raw import frame_bytes, frames_on_disk, recording_files
from ..session import noise_floor
from .cube import _batches, zero_filled
from .dsp import RangeDoppler

POINT_DB = ABOVE_FLOOR_DB      # dB above the floor to keep a cell (6): later steps can raise it with `db`
KEEP = (RANGE_AXIS >= MIN_RANGE)[:, None] & (np.arange(N_DOPPLER) != N_DOPPLER // 2)[None, :]


def points_path(person, scenario):
    return os.path.join(recording_dir(person, scenario), 'points.npz')


def load_points(person, scenario):
    """dict of the points.npz arrays"""
    with np.load(points_path(person, scenario)) as d:
        return {k: d[k] for k in d.files}


def select(spectrum, floor_db, keep):
    """
    :param spectrum: (n, tx, rx, range, Doppler) complex torch tensor (RangeDoppler.spectrum)
    :param floor_db: (range, Doppler) torch tensor, dB
    :param keep: (range, Doppler) bool torch tensor
    :return: frame offsets in the batch, range bins, Doppler bins, dB, and (k, 12) complex values of the kept cells
    """
    n = spectrum.shape[0]
    x = spectrum.reshape(n, 12, *spectrum.shape[-2:])                      # (n, 12, range, Doppler)
    db = 10 * torch.log10((x.real ** 2 + x.imag ** 2).mean(1) + 1e-30)     # as the cube
    f, r, d = torch.nonzero((db > floor_db + POINT_DB) & keep, as_tuple=True)
    return f, r, d, db[f, r, d], x[f, :, r, d]


def write_points(person, scenario, overwrite=False, device=None):
    """the points of one recording; skipped when points.npz and points_info.json exist (unless overwrite)"""
    out = points_path(person, scenario)
    info_file = os.path.join(recording_dir(person, scenario), 'points_info.json')
    if not overwrite and os.path.exists(out) and os.path.exists(info_file):
        print(f'{person}/{scenario}: points exist, skipped', flush=True)
        return
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rd = RangeDoppler(device)
    floor = torch.as_tensor(noise_floor(person, scenario), dtype=torch.float32, device=device)
    keep = torch.as_tensor(KEEP, device=device)
    paths, offsets = recording_files(person, scenario)
    T = frames_on_disk(paths, offsets, frame_bytes())
    parts = {k: [] for k in ('frame', 'rbin', 'dbin', 'db', 'iq')}
    n_read, skipped = 0, []
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
        i0, frames = item
        x = torch.as_tensor(frames, device=device)
        spec = rd.spectrum(x)
        bad = (spec.flatten(1).abs().amax(1) == 0) | zero_filled(x)    # no data, or damaged by lost packets
        f, r, d, db, iq = select(spec, floor, keep)
        ok = ~bad[f]
        f, r, d, db, iq = f[ok], r[ok], d[ok], db[ok], iq[ok]
        iq = iq / iq.abs().amax(1, keepdim=True).clamp_min(1e-30)
        parts['frame'].append((f + i0).cpu().numpy().astype(np.int32))
        parts['rbin'].append(r.cpu().numpy().astype(np.uint8))
        parts['dbin'].append(d.cpu().numpy().astype(np.uint8))
        parts['db'].append(db.cpu().numpy().astype(np.float16))
        parts['iq'].append(torch.view_as_real(iq).cpu().numpy().astype(np.float16))
        skipped += (i0 + torch.nonzero(bad).flatten().cpu().numpy()).tolist()
        n_read += len(frames)
        t_compute += time.time() - t
    arrays = {k: np.concatenate(v) if v else np.empty((0,) if k != 'iq' else (0, 12, 2)) for k, v in parts.items()}
    order = np.argsort(arrays['frame'], kind='stable')            # batches arrive in order; keep it guaranteed
    arrays = {k: v[order] for k, v in arrays.items()}
    tmp = out[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    total = time.time() - t_start
    n = len(arrays['frame'])
    info = dict(frames=T, frames_read=n_read, frames_skipped=skipped, points=n,
                points_per_frame=round(n / max(n_read - len(skipped), 1), 1), point_db=POINT_DB,
                min_range_m=MIN_RANGE, megabytes=round(os.path.getsize(out) / 1e6, 1), seconds=round(total, 1),
                waiting_for_data_s=round(t_wait, 1), compute_s=round(t_compute, 1), device=str(device))
    with open(info_file, 'w') as fp:
        json.dump(info, fp)
    print(f'{person}/{scenario}: {n_read} frames read, {len(skipped)} skipped, {n} points '
          f'({info["points_per_frame"]}/frame, {info["megabytes"]} MB); {total:.0f} s = {1e3 * total / max(T, 1):.1f} '
          f'ms/frame, waiting for data {t_wait:.0f} s, compute {t_compute:.0f} s', flush=True)
