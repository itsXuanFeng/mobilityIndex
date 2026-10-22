# Scope: mobility metrics from the ceiling FMCW radar (January 2026)

Goal: compute 10 mobility metrics in 3 categories from the FMCW radar, validated against MoCap. Only the January 2026 campaign is used, because it has MoCap.

## What we have

**Radar.** TI IWR6843AOP (antenna-on-package), mounted on the ceiling and looking down. 3 TX (time-multiplexed) × 4 RX = 12 virtual channels in a 2-D array, so we get both angles.

| | |
|---|---|
| Frame rate | 20 frames/s (100 chirps per TX per frame) |
| Range | 4.4 cm per bin; bins 0–127 cover 0–5.6 m |
| Doppler | ±2.7 m/s, 4.2 cm/s per bin |
| Raw data | format and missing data: `docs/raw_data_january.md` |

Because the radar looks down, **range ≈ distance from the ceiling to the top of the body (≈ height)**, and the **Doppler sign gives up/down**: positive = moving away from the radar.

**Ground truth.** In `/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/January_2026/Measurement Campaign Januari 2026/`:
- `MOCAP_playback/`: Captury markerless skeleton at 50 Hz, joint positions in mm, UTC timestamps in µs.
- `timestamps/`: start and end of each scripted activity (epoch seconds, UTC), 38 per scenario. We use these original timestamps as they are.
- `references_FMCW_MOCAP.png`: radar position relative to the MoCap origin.

**Known data issues**
- MoCap shots match radar scenarios by time, not by number; for example, scenario_4 was recorded before scenario_3.
- A shot holds several tracks: the subject (largest file) plus ghost tracks. Some shots have two large tracks (e.g. person_5, shots 5–6).
- MoCap starts 10–50 s after the first label, so the first activities are often not covered.
- The radar start time is logged only to the second, in local time (CET).
- MoCap is likely unreliable when the person lies under a blanket.

**Findings so far**
- Processing all 12 channels costs about 10 ms/frame, against about 184 ms/frame to read the raw data. Extra channels are practically free.
- Averaging the 12 channels halved the frame-to-frame jitter of the noise floor (0.98 → 0.44 dB; measured on August data).
- Transitions are visible in range and Doppler (person_1/scenario_1):

  | Event | Doppler sign | Nearest moving reflector |
  |---|---|---|
  | Sit down on chair | + (away) | 1.45 → 2.02 m |
  | Stand up from chair | − (towards) | 1.93 → 1.45 m |

  Peak speed was about 1 m/s, well within ±2.7 m/s.
- A person who keeps still is invisible: static reflections are removed per frame. Posture between transitions has to be inferred.
- The radar log gives its start time only to the second, so radar and MoCap need fine time alignment.

## Approach

Read the raw data once and produce a few per-frame radar signals. Turn them into per-frame states: activity level, active/inactive, and posture (lying, sitting, standing). Derive all 10 metrics from those states with fixed rules, so the metrics stay consistent with each other.

Radar products, all from one pass:
1. **Doppler profile** (T × 128): 12-channel power average, without range.
2. **Moving-energy range profile** (T × 128): approximately a time-height map.
3. **Angle / 3-D position** of the moving reflectors, from the AOP 2-D array. Needs the virtual-array positions and per-channel phase signs from TI's AOP antenna geometry, checked against MoCap.

### Category 1: total movement volume
*Total activity energy, active time, inactive time*

- **Signal:** Doppler profile (1) with the noise floor subtracted; ignore |v| < ~0.1 m/s to exclude residual clutter and breathing.
- **Distance correction:** received power falls steeply with distance, about 20 dB between 1.4 m and 4.5 m. Correct it with the range from (2), or use measures that don't depend on signal strength, such as Doppler spread or the number of bins above the floor.
- **Active/inactive:** threshold per epoch (e.g. 1 s), calibrated against MoCap.
- **Ground truth:** MoCap joint speeds per epoch: a kinetic-energy proxy, and a speed threshold for active.

### Category 2: activity bout structure
*Number of bouts, mean and variance of bout duration, fragmentation index*

- **Signal:** the active/inactive sequence from Category 1; no range needed.
- **Method:** bout rules: smoothing with hysteresis, a minimum bout length, and merging of short gaps. Fragmentation index, e.g. bouts per active hour, or the active→inactive transition probability.
- **Ground truth:** the same rules applied to the MoCap active/inactive sequence.
- **Caveat:** the recordings follow a script, so bout structure is imposed by the instructions. Validate mainly at the epoch level.

### Category 3: postural transitions
*Peak Doppler during transition, sit-to-stand, stand-to-sit, lie-to-sit counts*

- **Signal:** Doppler (1) for the motion burst and its sign; range (2) and angle (3) for height before and after.
- **Method:**
  1. Detect a transition as a burst of vertical motion.
  2. Its direction comes from the Doppler sign and the change in height.
  3. Its type comes from the start and end heights: bed, seated, standing.
  4. Carry the posture forward from one transition to the next, because a still person is invisible.
- **Off-centre positions:** use the angle to convert slant range to height.
- **Peak Doppler:** max |v| in the event. It is speed along the line to the radar, so compare it with MoCap speed along that same line.
- **Ground truth:** posture from the MoCap skeleton (head and pelvis height, trunk and knee angles) and the activity labels. Use the labels alone for lie-to-sit (blanket occlusion).

## Ground truth and evaluation

1. **Alignment:** pair shots with scenarios by time and select the subject's track. Align coarsely from the radar log start time, then cross-correlate radar motion with MoCap speed along the line to the radar.
2. **Coordinates:** radar position from the floor plan; refine the ceiling height with the data.
3. **Per-frame labels from MoCap:** activity level, active/inactive, posture, transitions.
4. **Evaluation**, split by person (leave-subjects-out):
   - per epoch: AUC and F1;
   - per event: transitions matched within ±1 s, F1;
   - per recording, on the 10 metrics: mean absolute error and bias in the metric's own units. Where radar and MoCap measure in different units (total activity energy), Spearman correlation across recordings.

## Steps

1. Alignment and MoCap ground truth (everything depends on this).
2. Radar pass producing products 1–2 from all 12 channels.
3. Rule-based metrics for all 3 categories; learned per-frame models only if needed.
4. AOP angle processing (product 3) for Category 3 and for rejecting other moving objects.
