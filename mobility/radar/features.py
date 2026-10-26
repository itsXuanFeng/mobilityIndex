"""
Per-frame radar features from the range-Doppler cube.

To add a feature, write a function of a `Chunk` (a block of frames with the floor already removed) that returns one
value per frame, and decorate it with @feature. It is computed for every recording the next time a session is loaded
(cached features are extended, not recomputed).

    @feature('my_feature', log=True)
    def my_feature(c):
        return (c.excess * something).sum((1, 2))
"""
from dataclasses import dataclass, field

import numpy as np

from ..config import (ABOVE_FLOOR_DB, COVERAGE_RADIUS, FLOOR_PERCENTILE, MIN_RANGE, MIN_SPEED, RANGE_AXIS,
                      VELOCITY_AXIS)


@dataclass
class Feature:
    name: str
    fn: callable
    log: bool      # compare / threshold it on a log scale (power, counts), or as is (velocities, ranges)
    doc: str


FEATURES = {}


def feature(name, log=True):
    def register(fn):
        FEATURES[name] = Feature(name, fn, log, (fn.__doc__ or '').strip())
        return fn
    return register


# cells that can hold a moving person: beyond the TX-RX leakage, faster than clutter / breathing
MOVING = (RANGE_AXIS >= MIN_RANGE)[:, None] & (np.abs(VELOCITY_AXIS) >= MIN_SPEED)[None, :]


@dataclass
class Chunk:
    """a block of frames: dB, linear power, the recording's noise floor, and the floor-removed moving power"""
    db: np.ndarray                     # (n, range, Doppler)
    floor_db: np.ndarray               # (range, Doppler)
    p: np.ndarray = field(init=False)
    excess: np.ndarray = field(init=False)   # power above the floor, moving cells only, >= 0
    on: np.ndarray = field(init=False)       # cells more than ABOVE_FLOOR_DB above their floor (moving cells only)

    def __post_init__(self):
        self.p = 10 ** (self.db / 10)
        self.excess = np.clip(self.p - 10 ** (self.floor_db / 10), 0, None) * MOVING
        self.on = (self.db > self.floor_db + ABOVE_FLOOR_DB) & MOVING


def noise_floor(cube, step=10):
    """noise floor per (range, Doppler) cell in dB: a low percentile over time (the person is still or away much of
    the time), on every step-th frame"""
    return np.nanpercentile(np.asarray(cube[::step], np.float32), FLOOR_PERCENTILE, axis=0)


def compute_features(cube, names=None, chunk=2000, floor_db=None):
    """
    :param cube: (T, range, Doppler) dB, e.g. mobility.radar.cube.load_cube(...)
    :return: {name: (T,) float64}, NaN for frames without data; and the noise floor (range, Doppler) in dB
    """
    names = list(FEATURES) if names is None else names
    floor_db = noise_floor(cube) if floor_db is None else floor_db
    T = len(cube)
    out = {n: np.full(T, np.nan) for n in names}
    for s in range(0, T, chunk):
        db = np.asarray(cube[s:s + chunk], np.float32)
        no_data = np.isnan(db).all((1, 2))
        c = Chunk(db, floor_db)
        for n in names:
            v = np.asarray(FEATURES[n].fn(c), float)
            v[no_data] = np.nan
            out[n][s:s + len(db)] = v
    return out, floor_db


# ---------------------------------------------------------------- Category 1: how much is moving

@feature('p_sub')
def p_sub(c):
    """moving power above the noise floor, summed over range and Doppler"""
    return c.excess.sum((1, 2))


# range^4 weight, zero beyond the radar coverage: there the subject never is, and the weight would amplify noise
R4 = (np.maximum(RANGE_AXIS, 0.5) ** 4 * (RANGE_AXIS <= COVERAGE_RADIUS))[:, None]


@feature('p_r4')
def p_r4(c):
    """p_sub within COVERAGE_RADIUS, each range bin weighted by range^4 (received power falls ~1/r^4):
    distance-corrected movement energy"""
    return (c.excess * R4).sum((1, 2))


@feature('n_cells')
def n_cells(c):
    """number of moving cells more than ABOVE_FLOOR_DB above their floor; insensitive to absolute power levels"""
    return c.on.sum((1, 2))


@feature('v_rms', log=False)
def v_rms(c):
    """RMS radial speed (m/s) of the cells above the floor; 0 when none is"""
    dop = (c.excess * c.on).sum(1)
    return np.sqrt((dop * VELOCITY_AXIS ** 2).sum(1) / np.maximum(dop.sum(1), 1e-30))


# ---------------------------------------------------------------- direction, height and speed (Category 3, alignment)

@feature('v_centroid', log=False)
def v_centroid(c):
    """signed mean radial velocity (m/s) of the moving power; + = away from the radar"""
    dop = c.excess.sum(1)
    return (dop * VELOCITY_AXIS).sum(1) / np.maximum(dop.sum(1), 1e-30)


@feature('r_near', log=False)
def r_near(c, db_above=12, min_cells=2, min_total=6):
    """range (m) of the nearest clearly moving reflector, ~ distance from the radar to the top of the moving body
    part; NaN when fewer than min_total cells are db_above dB above the floor"""
    strong = (c.db > c.floor_db + db_above) & MOVING
    per_range = strong.sum(2)
    first = np.argmax(per_range >= min_cells, axis=1)
    ok = (per_range.sum(1) >= min_total) & (per_range.max(1) >= min_cells)
    return np.where(ok, RANGE_AXIS[first], np.nan)


@feature('v_peak', log=False)
def v_peak(c, db_above=12, min_total=6):
    """largest |radial velocity| (m/s) of the cells db_above dB above the floor, the same clearly moving cells as
    r_near; NaN when fewer than min_total cells are. The first version's peak Doppler (Category 3): it reads high,
    because strong reflections leak power into neighbouring Doppler bins; the main method uses the angle features
    v_q900 ... v_q995 instead"""
    strong = (c.db > c.floor_db + db_above) & MOVING
    vmax = (strong * np.abs(VELOCITY_AXIS)).max((1, 2))
    return np.where(strong.sum((1, 2)) >= min_total, vmax, np.nan)
