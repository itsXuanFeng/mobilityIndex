"""
Direction and 3-D position of the points of the angle pass (mobility.radar.points).

Virtual array (IWR6843AOP), in half-wavelength units, from the colleague's localisation pipeline
(Sentigrate/distrimuse-ds-shared/fmcw_localization/fmcw_pipeline.py, from TI's array drawing):
    TX0 (0, 0), TX1 (2, -2), TX2 (0, -2)        RX0 (1, 0), RX1 (1, 1), RX2 (0, 0), RX3 (0, 1)       (a, b)
    virtual channel k = TX (k // 4) x RX (k % 4), position TX + RX; RX0 and RX2 have a 180 degree phase offset.
Chirp k is sent by TX k, 150 us apart, so a moving reflector also turns the phase between TXs: in Doppler bin d
(signed), TX k lags by 2 pi d k / (3 N_DOPPLER). Both are corrected in `channels`.

Direction cosines (u, v) along the array axes a and b; w = sqrt(1 - u^2 - v^2) along the boresight, which points
down. Two estimators:
    phase   least-squares phase difference between all channel pairs one half-wavelength apart (single snapshot,
            no grid; ESPRIT on one snapshot). Fast; assumes one dominant reflector per cell.
    beam    the peak of the beamformer over a grid of (u, v). Slower; a check on `phase`.

In the room (MoCap frame, Y up), the radar is at RADAR_XYZ, and its array axes are rotated by RADAR_YAW_DEG about the
vertical (mirrored when RADAR_MIRROR), fitted against MoCap (mobility.align.fit_radar_orientation,
scripts/fit_radar_orientation.py):
    x, z = radar + r (u a_room + v b_room),    y = radar height - r w
"""
import numpy as np

from ..config import N_DOPPLER, RADAR_MIRROR, RADAR_XYZ, RADAR_YAW_DEG, RANGE_AXIS

TX_POS = np.array([[0, 0], [2, -2], [0, -2]], float)
RX_POS = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], float)
RX_SIGN = np.array([-1, 1, -1, 1], float)
VA_POS = (TX_POS[:, None, :] + RX_POS[None, :, :]).reshape(12, 2)       # (12, [a, b])
N_TX = len(TX_POS)


def _pairs(step):
    """(reference, shifted) channel indices one half-wavelength apart along `step`"""
    ref, shf = [], []
    for i, p in enumerate(VA_POS):
        j = np.flatnonzero((VA_POS == p + step).all(1))
        if len(j):
            ref.append(i)
            shf.append(j[0])
    return np.array(ref), np.array(shf)


PAIRS_A, PAIRS_B = _pairs([1, 0]), _pairs([0, 1])


def channels(points):
    """(N, 12) complex64 channel vectors of the points (dict from load_points), corrected for the RX phase offsets
    and the TDM Doppler phase"""
    iq = points['iq'].astype(np.float32)
    x = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)
    d = points['dbin'].astype(np.float32) - N_DOPPLER // 2                  # signed Doppler bin
    tx = np.repeat(np.arange(N_TX), len(RX_POS))
    x *= np.tile(RX_SIGN, N_TX)[None, :]
    x *= np.exp(-2j * np.pi * d[:, None] * tx[None, :] / (N_TX * N_DOPPLER)).astype(np.complex64)
    return x


def direction_phase(x):
    """(u, v) per channel vector: least-squares phase difference over the pairs one half-wavelength apart"""
    def cosine(pairs):
        ref, shf = pairs
        return np.angle((np.conj(x[:, ref]) * x[:, shf]).sum(1)) / np.pi
    return cosine(PAIRS_A), cosine(PAIRS_B)


def direction_beam(x, n=65, chunk=20000):
    """(u, v) per channel vector: peak of the beamformer |a(u, v)^H x| over a grid of n x n direction cosines inside
    the unit circle"""
    g = np.linspace(-1, 1, n)
    uu, vv = [a.ravel() for a in np.meshgrid(g, g, indexing='ij')]
    inside = uu ** 2 + vv ** 2 <= 1
    uu, vv = uu[inside], vv[inside]
    steer = np.exp(1j * np.pi * (np.outer(VA_POS[:, 0], uu) + np.outer(VA_POS[:, 1], vv))).astype(np.complex64)
    best = np.concatenate([np.argmax(np.abs(x[s:s + chunk] @ steer.conj()), axis=1)
                           for s in range(0, len(x), chunk)]) if len(x) else np.array([], int)
    return uu[best], vv[best]


def to_room(r, u, v, yaw_deg=RADAR_YAW_DEG, mirror=RADAR_MIRROR, radar_xyz=RADAR_XYZ):
    """(x, y, z) in the MoCap frame from range r and direction cosines (u, v); NaN outside the unit circle"""
    r, u, v = np.asarray(r, float), np.asarray(u, float), np.asarray(v, float)
    w = np.sqrt(np.clip(1 - u ** 2 - v ** 2, 0, None))
    w[u ** 2 + v ** 2 > 1] = np.nan
    yaw = np.deg2rad(yaw_deg)
    a = np.array([np.cos(yaw), np.sin(yaw)])                               # array axis a in the (x, z) plane
    b = np.array([-np.sin(yaw), np.cos(yaw)]) * (-1 if mirror else 1)     # axis b, perpendicular
    x = radar_xyz[0] + r * (u * a[0] + v * b[0])
    z = radar_xyz[2] + r * (u * a[1] + v * b[1])
    return x, radar_xyz[1] - r * w, z


def localise(points, method='phase', **kw):
    """the points with their direction cosines and room position: dict of arrays u, v, x, y, z added"""
    x = channels(points)
    u, v = direction_phase(x) if method == 'phase' else direction_beam(x)
    px, py, pz = to_room(RANGE_AXIS[points['rbin']], u, v, **kw)
    return {**points, 'u': u, 'v': v, 'x': px, 'y': py, 'z': pz}


def nearest_reflector(points, floor_db, method='phase', db_above=12, min_cells=2, min_total=6):
    """
    per frame, the nearest clearly moving reflector, the same one as the feature r_near (cells db_above dB above the
    floor, moving; the first range bin with min_cells of them, frames with at least min_total): its range and the
    power-weighted mean direction cosines of its cells
    :return: dict of arrays frame, r, u, v, cells
    """
    from .features import MOVING
    strong = (points['db'].astype(np.float32) > floor_db[points['rbin'], points['dbin']] + db_above) & \
        MOVING[points['rbin'], points['dbin']]
    p = {k: v[strong] for k, v in points.items()}
    if not len(p['frame']):
        return dict(frame=np.array([], int), r=np.array([]), u=np.array([]), v=np.array([]), cells=np.array([], int))
    x = channels(p)
    u, v = direction_phase(x) if method == 'phase' else direction_beam(x)
    w = 10 ** (p['db'].astype(np.float32) / 10)
    df = {k: [] for k in ('frame', 'r', 'u', 'v', 'cells')}
    bounds = np.flatnonzero(np.diff(p['frame'])) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(p['frame'])]):
        if e - s < min_total:
            continue
        rb = p['rbin'][s:e]
        bins, counts = np.unique(rb, return_counts=True)
        ok = bins[counts >= min_cells]
        if not len(ok):
            continue
        m = rb == ok[0]
        ww = w[s:e][m]
        df['frame'].append(p['frame'][s])
        df['r'].append(RANGE_AXIS[ok[0]])
        df['u'].append(np.sum(u[s:e][m] * ww) / ww.sum())
        df['v'].append(np.sum(v[s:e][m] * ww) / ww.sum())
        df['cells'].append(int(m.sum()))
    return {k: np.asarray(v) for k, v in df.items()}


# ---- per-frame features of the person (Category 3: posture and peak Doppler) ----

STRONG_DB = 12             # dB above the floor: clearly visible points, as for r_near and v_peak
PERSON_RADIUS_M = 1.0      # points this close (horizontally) to the frame's centre belong to the person
MIN_PERSON_POINTS = 3      # fewer: the person isn't seen in the frame
TOP_QUANTILE = 0.9         # top of the body: this quantile of the person's point heights (robust to stray points)
# peak Doppler: power-weighted quantiles of the person's moving speeds; which one is calibrated against MoCap
PEAK_QUANTILES = (0.9, 0.95, 0.98, 0.99, 0.995)
PEAK_FEATURES = tuple(f'v_q{round(q * 1000)}' for q in PEAK_QUANTILES)
ANGLE_FEATURES = ('h_top', 'x_person', 'z_person', 'n_person', 'n_slow') + PEAK_FEATURES


def frame_features(points, floor_db, n_frames):
    """
    per frame, from the angle-pass points, the person as the radar sees them, whether moving or not (the slow Doppler
    bins hold breathing and small movements, so a still person is usually visible):
        h_top       height of the top of the body (m): TOP_QUANTILE of the heights of the person's points
        x_person, z_person   position of the person (m, MoCap frame): median of all clearly visible points
        n_person    number of the person's points;  n_slow: of them, slower than MIN_SPEED (still)
        v_q900 ... v_q995    speed of the person (m/s, along the line to the radar): the power-weighted quantile
                    PEAK_QUANTILES of the |velocity| of their moving points. A high quantile, not the fastest cell:
                    a strong reflection leaks power into neighbouring Doppler bins, and the fastest cell above the
                    floor sits on that leakage (docs/category3.md, Peak Doppler)
    Clearly visible: STRONG_DB above the floor, inside the radar coverage and the room, between the floor and the
    radar. The person: within PERSON_RADIUS_M of the position. NaN (counts 0) where fewer than MIN_PERSON_POINTS.
    The caregiver and operator would also be seen: their periods are outside the evaluation window.
    """
    import pandas as pd
    from ..config import COVERAGE_RADIUS, MIN_SPEED, ROOM_MAX_X, VELOCITY_AXIS
    out = {k: np.full(n_frames, np.nan) for k in ANGLE_FEATURES}
    out['n_person'][:] = out['n_slow'][:] = 0
    strong = points['db'].astype(np.float32) > floor_db[points['rbin'], points['dbin']] + STRONG_DB
    p = localise({k: v[strong] for k, v in points.items()})
    ok = ((RANGE_AXIS[p['rbin']] <= COVERAGE_RADIUS) & (p['x'] < ROOM_MAX_X) & (p['y'] > -0.3)
          & (p['y'] < RADAR_XYZ[1]))
    speed = np.abs(VELOCITY_AXIS[p['dbin'][ok]])
    df = pd.DataFrame(dict(frame=p['frame'][ok], x=p['x'][ok], y=p['y'][ok], z=p['z'][ok], speed=speed,
                           power=10 ** (p['db'][ok].astype(np.float64) / 10), slow=speed < MIN_SPEED))
    if not len(df):
        return out
    centre = df.groupby('frame')[['x', 'z']].median()
    c = centre.loc[df.frame.values].values
    person = df[np.hypot(df.x.values - c[:, 0], df.z.values - c[:, 1]) <= PERSON_RADIUS_M]
    g = person.groupby('frame')
    n = g.size()
    n = n[n >= MIN_PERSON_POINTS]
    f = n.index.values
    out['n_person'][f] = n.values
    out['n_slow'][f] = g.slow.sum().loc[f].values
    out['h_top'][f] = g.y.quantile(TOP_QUANTILE).loc[f].values
    out['x_person'][f] = centre.x.loc[f].values
    out['z_person'][f] = centre.z.loc[f].values
    mv = person[~person.slow & person.frame.isin(f)].sort_values(['frame', 'speed'])
    if len(mv):
        frac = mv.groupby('frame').power.cumsum() / mv.groupby('frame').power.transform('sum')
        for q, name in zip(PEAK_QUANTILES, PEAK_FEATURES):
            vq = mv[frac.values >= q - 1e-9].groupby('frame').speed.first()
            out[name][vq.index.values] = vq.values
    return out
