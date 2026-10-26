# The angle pass (product 3)

The cube (`rd12_db.npy`) averages the 12 virtual channels, so it has range and Doppler but no direction: the radar's "height" so far is a slant range, which mixes height with horizontal distance. The angle pass reads the raw data a second time and keeps the per-channel complex values that carry the direction. This note covers what it stores, the check against MoCap on the 6 development recordings, and the campaign run. **Status: done for all 156 January recordings (2026-10-25 18:31 – 2026-10-26 01:28, log `output/angle_pass_january.log`); checked against MoCap on the 6 development recordings (2026-10-26).**

## Summary

- **The geometry is right.** The virtual-array layout and RX phase flips from a colleague's localisation pipeline, plus the TDM Doppler correction, put the radar's nearest moving reflector within **14 cm** (median, horizontally) of the nearest MoCap joint, on 16,096 frames of the 6 development recordings, one per furniture layout (13–15 cm per recording).
- **Height from the angle is much better than from range.** Median height error 9 cm with the angle, against 53 cm from range alone (what Categories 1–3 use). 2–3 m off the axis: 15 cm against 141 cm.
- **The radar's orientation is simple:** its array axes lie along MoCap x and z (each recording fitted alone: yaw between −1° and +1°; never mirrored).
- **The full run** took 6.9 h for 156 recordings (25 GB), network-bound as the cube pass; the GPU was not needed.

## What is stored

`scripts/angle_pass.py` → `mobility/radar/points.py`, per recording in `output/<person>/<scenario>/`:

| File | Content |
|---|---|
| `points.npz` | one row per kept cell ("point"), sorted by frame: `frame` (int32, = cube row), `rbin`, `dbin` (uint8, as in the cube), `db` (float16, 12-channel mean power as in the cube), `iq` (float16, N × 12 × 2: the complex value per virtual channel divided by the largest of the 12) |
| `points_info.json` | frames read and skipped, points, size, timing, device |

- **Kept cells:** range ≥ 0.3 m, any Doppler bin except zero velocity, more than 6 dB above the cell's noise floor. The floor is the recording's own, from the cube (`mobility.session.noise_floor`), so the cube and its features must exist first. Slow bins (breathing, small movements, ~0.04 m/s) are kept on purpose: they may locate a person who keeps still. About 280 points per frame on average (median frame: about 100).
- **Only the measurement is stored.** No angle method, RX phase correction, TDM correction or orientation is applied, so all of them can change later without re-reading the raw data. Normalising each point's 12 values by their largest keeps phases and relative amplitudes exact enough in float16.
- **Skipped frames:** without data or damaged by lost packets, the frames that are NaN in the cube.

## From points to positions

`mobility/radar/angle.py`, all downstream and cheap (seconds per recording):

1. **Channel vectors** (`channels`): RX0 and RX2 sign-flipped (180° hardware offset); TX *k* rotated back by 2π·d·k / (3·128) for Doppler bin *d*, because chirp *k* is sent by TX *k* 150 µs later and a moving reflector turns the phase in between.
2. **Direction cosines** (u, v) along the array axes:
   - `phase` (default): least-squares phase difference over all channel pairs half a wavelength apart (6 per axis). One dominant reflector per cell is assumed.
   - `beam`: peak of the beamformer over a 65 × 65 grid; a check on `phase`.
3. **Room position** (`to_room`): from the radar at `RADAR_XYZ` looking down, rotated by `RADAR_YAW_DEG` (mirrored if `RADAR_MIRROR`): x, z = radar + r·(u, v) rotated; height = radar height − r·√(1 − u² − v²).

Virtual array, in half wavelengths (a, b), from `Sentigrate/distrimuse-ds-shared/fmcw_localization/fmcw_pipeline.py`: TX0 (0, 0), TX1 (2, −2), TX2 (0, −2); RX0 (1, 0), RX1 (1, 1), RX2 (0, 0), RX3 (0, 1). Channel *k* = TX *k* // 4 × RX *k* % 4; chirp *k* is sent by TX *k* (`config.mmwave.json`).

## Check against MoCap

`scripts/fit_radar_orientation.py` on the 6 development recordings, one per layout: person_1/scenario_1, person_10/scenario_2 (chair 2.6 m off-axis), person_5/scenario_3, person_12/scenario_5, person_15/scenario_4, person_22/scenario_6 (log `output/fit_radar_orientation.log`):
- **Radar:** per frame, the nearest clearly moving reflector, the same as feature `r_near`, with the power-weighted direction of its cells.
- **MoCap:** the joint nearest to the radar (head, shoulders, wrists, knees), frames where a joint moves faster than 0.3 m/s and the two ranges agree within 15 cm (16,096 frames).
- **Fit:** the rotation about the vertical and the mirror that best map the measured directions onto the MoCap ones (`mobility.align.fit_radar_orientation`).

| | `phase` | `beam` |
|---|---|---|
| fitted yaw, mirror (pooled) | 0.0°, no | 1.0°, no |
| median direction error (direction cosines) | 0.068 | 0.070 |
| median horizontal error | **14 cm** (per recording 13–15) | 14 cm |
| median height error, angle (bias) | **9 cm** (−1) | 9 cm (−4) |
| median height error, range alone (bias) | 53 cm (−53) | 53 cm (−53) |

By the joint's distance from the radar axis (`phase`, medians):

| Distance | Frames | Horizontal error | Height error, angle | Height error, range alone |
|---|---|---|---|---|
| < 0.5 m | 3,375 | 12 cm | 6 cm | 7 cm |
| 0.5–1 m | 3,537 | 11 cm | 7 cm | 22 cm |
| 1–1.5 m | 2,296 | 14 cm | 8 cm | 54 cm |
| 1.5–2 m | 3,157 | 14 cm | 11 cm | 102 cm |
| 2–3 m | 2,534 | 17 cm | 15 cm | 141 cm |
| 3–5 m | 1,142 | 25 cm | 25 cm | 277 cm |

- The errors include the mismatch between "nearest reflector" and "nearest joint" (a hand vs a wrist marker, clothing), so the angle itself is probably better than this.
- The two estimators agree; `phase` is about 200 times faster (0.04 s against 7.6 s for 200,000 points), so it is the default.
- Each recording fitted alone gives the same orientation: −1°, +0.5°, 0°, +1°, −1°, +0.5° (person_1, 10, 5, 12, 15, 22), never mirrored. The radar was not turned between the sessions; `RADAR_YAW_DEG = 0`, `RADAR_MIRROR = False` in `mobility/config.py` stand.

![person_10/scenario_2, angle check](../output/plots/angle/person_10_scenario_2_angle.png)

x, z and height of the radar's nearest moving reflector (orange) and the nearest MoCap joint (blue), person_10/scenario_2. Height from range alone (grey) drops below the floor whenever the person is away from the axis.

## Running it on the campaign

Run on 2026-10-25 18:31 – 2026-10-26 01:28 on `barabas` with the first command below (`--device cpu --threads 8`; both GPUs were busy with other jobs). Result: 156 of 156 recordings, no failures; 1,601,282 frames read (the 26,231 missing frames are those missing from the cube too: person_101/scenario_100 frames 0–26,214, person_23/scenario_5 frames 858–873); 120 frames skipped as damaged, the same as in the cube; 447 M points, 25 GB.

```bash
PY=/cw/dtaiexp/NoCsBack/2026-Xuan/miniforge3/envs/distrimuse/bin/python3
nohup $PY scripts/angle_pass.py --device cpu --threads 8 > output/angle_pass_january.log 2>&1 &
# then the orientation check (done 2026-10-26, results above):
$PY scripts/fit_radar_orientation.py --only person_1/scenario_1 person_5/scenario_3 person_10/scenario_2 \
    person_12/scenario_5 person_15/scenario_4 person_22/scenario_6
```

| | Value (estimated from the 3 test recordings, before the run) |
|---|---|
| time | 14.3–14.7 ms per frame, ~2/3 of it waiting for data → about 6.5 h for 1.63 M frames |
| size | 15.8 kB per frame → about 26 GB (the cubes take 51 GB; 682 GB free on 2026-10-25) |
| device | CPU with 8 threads keeps up with the network (compute ~40% of the time); a free GPU works too (`--device cuda`) |
| already done | the 3 test recordings: they are skipped (`--overwrite` to redo) |
| needs | the cube and its noise floor (`features.npz`) of every recording: all 156 have them (computed on 2026-10-25 for the 6 without MoCap) |

Recordings run one after the other: the network is the limit, as for the cube pass. A failed recording is reported at the end and the others continue; re-running the same command does only the missing ones.

## Not done yet

- **Used by Category 3** (2026-10-26): the person's top-of-body height and position per frame (`frame_features`, cached in `angle_features.npz`) give the posture per epoch, also while still, and the transitions ([category3.md](category3.md)). Still to do: separating the subject from a caregiver, to stop excluding the caregiver visit.
- **One reflector per cell.** `phase` assumes it; where two body parts share a range-Doppler cell, `beam` (or a multi-snapshot method over neighbouring Doppler bins, as the colleague's MUSIC/ESPRIT) may do better.
- **Checked on the 6 development recordings only.** One per layout, so the orientation is confirmed for every layout; the other 150 recordings are not checked one by one.
