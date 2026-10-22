"""Paths, radar constants, room geometry and thresholds used across the package."""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- data (January 2026 campaign, read-only) ----
CAMPAIGN = '/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/January_2026/Measurement Campaign Januari 2026'
FMCW_ROOT = f'{CAMPAIGN}/FMCW'
MOCAP_ROOT = f'{CAMPAIGN}/MOCAP_playback'
LABEL_ROOT = f'{CAMPAIGN}/timestamps'

# everything this package writes: radar cubes, cached features, evaluation tables, plots
OUTPUT_ROOT = os.environ.get('MOBILITY_OUTPUT', os.path.join(REPO, 'output'))

# ---- radar (mobility/io/config.mmwave.json) ----
FPS = 20                     # frames per second
N_RANGE = 128                # range bins kept (0-5.6 m)
N_DOPPLER = 128
RANGE_BIN = 0.04395          # m per range bin
VELOCITY_BIN = 0.04340       # m/s per Doppler bin, ±2.78 m/s
RANGE_AXIS = np.arange(N_RANGE) * RANGE_BIN
VELOCITY_AXIS = (np.arange(N_DOPPLER) - N_DOPPLER // 2) * VELOCITY_BIN   # + = moving away from the radar

# ---- per-frame radar features ----
MIN_SPEED = 0.10             # m/s; slower Doppler bins are residual clutter, breathing, a person standing still
MIN_RANGE = 0.30             # m; the first range bins hold TX-RX leakage
FLOOR_PERCENTILE = 20        # noise floor per (range, Doppler) cell: this percentile over the recording
ABOVE_FLOOR_DB = 6           # a cell "is on" when this far above its floor

# ---- geometry, in the MoCap frame (metres, Y up) ----
RADAR_XYZ = (-3.25, 2.45, 0.40)   # fitted with scripts/fit_radar_position.py; the floor plan agrees in x and z
ROOM_MAX_X = 1.0                  # x beyond this is the corridor
COVERAGE_RADIUS = 5.0             # subject CoG farther than this from the radar is outside radar coverage

# ---- epochs and ground truth ----
EPOCH_S = 1.0
MOCAP_ACTIVE_SPEED = 0.15         # MoCap: an epoch is active if the RMS joint speed exceeds this (m/s)
ALIGN_SEARCH_S = (-15.0, 15.0)    # fine time offset searched around the logged capture start
