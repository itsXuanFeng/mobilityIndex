# Calibration strategy: population and personal

How the parameters of Categories 1–3 are calibrated:
- **Today:** everything is calibrated leaving the evaluated person out (LOSO).
- **Proposed:** a personal step on top of LOSO, tested in three setups.

This page sets out the strategy and lists every parameter with how it is, or will be, calibrated. The experiment is implemented in `mobility/evaluation/personal.py` and `scripts/evaluate_personal_calibration.py`; its latest run (2026-11-19) is in `output/personal/` (log `output/personal_full.log`).

## Summary

- **Population first, then personal.**
  1. **Population (LOSO):** calibrate on the other 26 people, as now.
  2. **Personal:** adjust a few parameters to the resident, from the resident's own data.
- **What decides it.**
  - What *defines* a metric stays population-wide: what counts as active, what a bout is, the MoCap posture classes. This keeps the metrics comparable between residents and over time.
  - What *maps the radar onto* that definition may be personal. **Person** parameters belong to the body (stature, body size, reflectivity); **room** parameters belong to the room (bed and chair heights, where the person sits relative to the radar).
- **Personal means an adjustment, not a replacement.** A personal parameter starts from its LOSO value and is shifted by the person's own data. With little data, it stays close to the LOSO value.
- **In deployment** there are only two sources: LOSO, and a short guided session at installation in the resident's own room. That session gives both the person and the room parameters. The values are then frozen.
- **The experiment** compares three setups. The realistic one is LOSO plus a short labelled clip of the evaluated recording, which simulates the session.

## Why personal calibration

LOSO answers: *how well does it work for a new resident, without any setup?* That stays the main number for generalisation.

Some of the remaining errors are steady per person, not noise:
- **Active time:** −0.4 to +0.4 min per person.
- **Energy:** the converted radar energy is off by a steady amount per person (person_26 −19% in every recording).
- **Bout counts:** person_1 and person_25 have consistently fewer radar bouts.
- **Posture heights:** they depend on stature and on the furniture.

A short calibration in the resident's own room can remove such offsets. The second number to report is therefore: *how well does it work after a short setup?*

## Principles

1. **Definitions stay population-wide.**
   - **What this covers:** the MoCap reference rules (active threshold, bout factors, posture rule) are calibrated against the labels on other people, never on the evaluated person.
   - **Why:** a personal definition of "active" or of a bout would make a slow resident's thresholds lower. The calibration would then hide the decline we want to measure.
   - **The resident's speed:** their usual speed must never enter the definitions. It may enter only the radar's mapping, such as the signal level or the energy scale, and only against a reference (a timed walk), never by assuming the resident moves like everyone else.
2. **Personal parameters only correct the sensor:** the body, the placement relative to the radar, and the room.
3. **Rooms are all different.** There is no set of layouts to choose from. Room parameters come only from data recorded in that room.
4. **Shrink towards LOSO.** A personal value is LOSO + *w* × (personal estimate − LOSO). The weight *w* grows with the amount of personal data, and is itself calibrated on the training people, by simulating the clip on them.
5. **Freeze after enrolment.** The personal values are fixed after calibration. They are redone only on explicit events: furniture moved, the radar remounted, a new walking aid. Continuous adaptation would slowly absorb real changes in mobility.
6. **Evaluate on unseen data.** A personal parameter is never judged on data it was fitted on.

## Three setups (experiment on the January data)

| Setup | Population | Person parameters | Room parameters | Answers |
|---|---|---|---|---|
| **A. LOSO** | other 26 people | LOSO | LOSO | no setup at all |
| **B. LOSO + clip** | other 26 people | from the clip (labels only) | from the clip (labels only) | **the realistic one:** a short installation session |
| **C. LOSO + other recordings + clip** | other 26 people | from the person's other recordings (MoCap) | from the clip (labels only) | upper bound: is more personal data worth collecting? |

**Deployment** has only setup B: LOSO plus the installation session. Setup C has no counterpart. It uses about 25 min of MoCap-referenced data from the same person, far more than a session, so it is optimistic. The gap between B and C shows how much a longer or repeated enrolment could add.

Why the person's other recordings can't give the room parameters: in January, each person's recordings are in different furniture layouts (scenario = layout). They describe the person, but not the room of the evaluated recording.

**The clip** simulates the installation session inside the evaluated recording:
- **Content:** the first labelled occurrence of each session step, each interval tagged with its posture:
  - walking (`walk inside the main room`, `wander around in the room`): standing, active;
  - eating on the chair (`start performing eating movements`): sitting, inactive;
  - sitting on the edge of the bed: the gap after the first `sit down on the edge of the bed`, until the next label (about 2–9 s): sitting;
  - lying still in bed (`lie on your back again and lie still`, `press the button on the wearable`): lying.

  For the Category 3 standing height, only walking epochs within 2 m of the radar axis count (the session's "stand under the radar"; the radar's own position estimate). 2 m keeps at least 3 standing epochs in 96% of the recordings (1.5 m: 80%), where the radar height is within about 11 cm (the angle check in [angle.md](angle.md)).
- **Not continuous:** these steps are spread across the script. The clip is their union, a few seconds to a minute each.
- **Labels only:** the clip is fitted on the scripted labels (the Category 1 active / inactive classes, the clip interval's posture), as a caregiver's timestamps would give. MoCap isn't used in the clip, and the clip isn't restricted to epochs with MoCap: MoCap often starts after the first labels, which would empty the walking steps. Its epochs only need to be inside the script, without the caregiver, with radar data. A walking-speed reference (a stand-in for the session's timed walk, from MoCap) would be the one exception; it isn't used yet.

**Evaluation:**
- **Same epochs for all setups:** every setup is evaluated on the rest of the recording, including setup A, so the three are compared on the same data. Left out: the clip segments, and the postural-change labels right before and after a clip segment (for example sitting down before eating, or sitting up after the button press: their before or after posture is the level the clip fitted), each with a 2 s margin.
- **Paired per recording:** the difference of each setup to A, recording by recording, over all 148. Little data is left per recording (a few minutes), so per-recording errors are noisy, but the pairing removes most of that.
- **Scores**, as in each category:
  - Category 1: F1 of the active calls, active-time MAE, energy error and Spearman;
  - Category 2: count and duration errors, bout F1;
  - Category 3: posture F1, transition F1, posture-time errors.

**Each personal source fits only its own part.** In setup C, person parameters come from the other recordings and room parameters from the clip; the clip doesn't refit the person parameters. A parameter that depends on both (the radar active threshold) gets two steps: first the other recordings, then a room offset from the clip. In setup B, the clip fits everything.

**Caveats:**
- **The remainder is not the whole script.** The first walk, chair and bed periods go into the clip, so the evaluated part holds relatively more of the later activities (floor, second sit and stand).
- **Little data per person** (setup C): about 30 min in total, over different layouts; shrinkage handles overfitting.
- **Labels are not taps.** Scripted labels are more precise than a caregiver tapping a phone; a jitter of ±1 s on the clip boundaries can be added to test that.
- **The weights are chosen with nearly nested data.** For a left-out person p, each weight is chosen on the other persons q, each with its own LOSO base. That base was calibrated on everyone but q, so p's data is in it. The effect is one person in 26 on a population value; it isn't removed.

## Results: Category 1

Run of 2026-11-19 (`output/personal/`), 148 recordings. The clip holds 48 s per recording (median; 39–62), and the evaluated remainder 241 s (200–292). The errors are therefore smaller in absolute terms than in [category1.md](category1.md), which evaluates the whole recording. "Better / worse" is the share of recordings whose error went down / up against A; p is the Wilcoxon signed-rank test.

| Score | A. LOSO | B. LOSO + clip | C. LOSO + other recordings + clip |
|---|---|---|---|
| F1 of the active calls, median | 0.908 | 0.908 (p = 0.33) | 0.908 (p = 0.77) |
| active time: MAE, bias | 0.128 min, +0.063 | 0.133 min, +0.063 | 0.131 min, +0.058 |
| active time against A: better / worse, p | | 1% / 8%, p = 0.009 | 31% / 37%, p = 0.51 |
| energy: error, Spearman | 6.7%, 0.90 | 6.7%, 0.90 (= A by construction) | **4.8%, 0.95** |
| energy against A: better / worse, p | | | 64% / 36%, p < 0.001 |

Weights chosen on the training persons (number of the 27 left-out persons):

| Weight | 0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|
| threshold, B: clip shift | 23 | 4 | | |
| threshold, C: person's other recordings | 23 | 3 | 1 | |
| threshold, C: clip shift (room) | | 27 | | |
| energy factor, C: person's other recordings | | | | 27 |

- **The radar active threshold doesn't gain from personalisation.** The training persons mostly reject the clip shift (B: weight 0 for 23 of 27) and the person's own threshold (C: 0 for 23). Where B does shift it, it does slightly worse (8% of recordings worse, 1% better). The remaining active-time error is not a steady per-person offset of the threshold that a clip or other recordings can see.
- **Shrinkage matters in C.** With the person's own threshold used as is (the first run of this experiment), C's active-time MAE was 0.143 min, worse than A. Shrunk, it is 0.131 min, on par with A.
- **The energy does gain from a personal factor.** C's factor (weight 0.75 for every person) brings the error from 6.7% to 4.8%, and Spearman from 0.90 to 0.95. Per-person energy scales are real, but they come from MoCap on the person's other recordings, which deployment doesn't have. B can't reach them without an energy reference in the session (the timed walk).
- **Category 2 follows the setup's threshold.** In C, the radar bout entry is C's threshold, and the exit and confirmation levels are fitted on the person's other recordings around the shrunk person threshold. Those two levels are not shrunk themselves: Category 2 in C is worse than A (bout count error +0.38, p = 0.004; bout F1 −0.017, p < 0.001).

## The guided session (deployment)

About 5–10 minutes at installation. The caregiver taps a timestamp at each step; no MoCap is needed.

| Step | Calibrates |
|---|---|
| stand still under the radar, then near the chair (off-axis) | standing height (stature); off-axis check |
| walk a marked path at their own pace, out and back | radar active level; horizontal speed against the known distance and time |
| sit on their chair, still, 1 min | sitting height at the chair; inactive radar level |
| sit on the edge of the bed, lie down, lie still 1 min, sit up | sitting and lying heights at the bed; inactive level in bed (blanket) |
| stand up from the chair and from the bed, twice each | a check of the transitions (not fitted) |

From the first day(s), without labels, the rest positions locate the bed and chair. This also checks the session values: for example, the height distribution at the bed should match the lying height.

Too few bouts happen in the session to fit bout parameters (Category 2), so those stay LOSO. The radar entry threshold is the exception: it is personal, and the Category 2 exit and confirmation move with it, since they are offsets from it.

## Parameters per category

Strategies:
- **Fixed:** design, physics or hardware.
- **Installation:** the radar mounting and the room geometry, measured once per installation.
- **LOSO:** the other 26 people.
- **LOSO + person / LOSO + room:** LOSO value, then a personal adjustment.

For the personal ones, the column gives the source in setup B (clip), setup C (other recordings) and deployment.

### Shared (all categories)

| Parameter | Meaning | Calibration strategy |
|---|---|---|
| `RADAR_XYZ` | radar position (x, height, z) in the room frame | **Installation.** Fitted once on the 6 development recordings (`r_near` against MoCap). Deployment: measured at installation and checked with a walk under the radar. |
| `RADAR_YAW_DEG`, `RADAR_MIRROR` | orientation of the virtual array | **Installation.** Fitted once on the development recordings. |
| `COVERAGE_RADIUS`, `ROOM_MAX_X` | evaluation coverage | **Installation.** Room geometry. |
| `MIN_SPEED`, `MIN_RANGE` | moving-cell limits (0.1 m/s, 0.3 m) | **Fixed.** Breathing and TX–RX leakage. |
| `FLOOR_PERCENTILE`, `ABOVE_FLOOR_DB` | offline noise floor (20th percentile) and on-threshold (6 dB) | **Fixed.** |
| clutter map (20th pct, τ 2 min), SO-CFAR k (4.5 dB) | online detector | **Fixed.** It adapts to the room by itself; no personal setting. |
| `STRONG_DB`, `PERSON_RADIUS_M`, `MIN_PERSON_POINTS` | the person's points (12 dB, 1 m, ≥ 3 points) | **Fixed.** |
| `EPOCH_S` | epoch length (1 s) | **Fixed.** |
| time-alignment offset | radar vs MoCap clock | Experiment only; not needed in deployment. |

### Category 1: total movement volume

| Parameter | Meaning | Calibration strategy |
|---|---|---|
| MoCap active threshold | RMS joint speed above which an epoch is active (0.25 m/s) | **LOSO**, against the labels. It is the reference definition, never personal. |
| `MAX_MOCAP_RMS` | MoCap tracking-jump cut (3 m/s) | **Fixed**, set by hand. It acts on the 1 s RMS: smaller jumps pass and still add to the reference energy. |
| radar active threshold (`n_cells` / `n_det` level) | radar level above which an epoch is active | **LOSO + room and person.** It depends on the distance to the radar and on the body. The clip shift is the mean of how far the person's clip active (walking) and inactive (eating, the button, waiting) median levels lie from a reference: a location shift, so it doesn't need examples at the boundary. B: LOSO + *w* × the shift against the other persons' clips. C: LOSO + *w*<sub>own</sub> × (threshold fitted against MoCap on the person's other recordings − LOSO), then + *w* × the shift against the person's own other clips. Every weight is chosen on the training persons on active-time error (results below). Deployment: as B, from the session. |
| energy conversion factor | radar `e_move` → MoCap energy units (total MoCap ÷ total radar) | **LOSO + person**, setup C only. B: LOSO; the clip has no energy reference, so B's energy equals A's by construction. C: LOSO + *w*<sub>own</sub> × (factor on the person's other recordings − LOSO), *w*<sub>own</sub> chosen on the training persons on energy error. The person-level errors come mostly from how a person moves relative to a ceiling radar (the radial share of the motion, walking speed), not from body size or reflectivity: `e_body` is an unweighted mean over the person's points. So C's factor absorbs movement style, which deployment can't measure without MoCap. Candidate for B and deployment: a personal `v_horiz` scale from a timed walk (path length ÷ time); not in the experiment yet. |
| `e_move` weights | `e_body` + 1 × `v_horiz`² | **Fixed.** Physics. The free fit (0.97) that confirms it was done on the same 148 recordings, not leaving persons out. |
| radar level transform | the epoch level: log10 of the count, offset by the recording's 5th percentile (or its smallest positive value) | **Fixed**, but recording-specific: the offset needs the whole recording. For a continuous stream (deployment), a fixed transform such as log10(*n* + 1) should replace it before thresholds are made personal. It is nearly always an offset of 1 count here, so the results barely depend on it. |
| label classes, `LABEL_MARGIN_S` | active (whole-body movement) and inactive (arm-only tasks, waiting at least 2 s from a label) | **Fixed.** Definition (`mobility/labels.py`); they define the MoCap threshold and the clip's classes. |
| `v_horiz` smoothing | 0.5 s median, then 1 s mean | **Fixed.** |

### Category 2: activity bout structure

| Parameter | Meaning | Calibration strategy |
|---|---|---|
| entry threshold | start of a bout: the Category 1 active threshold of each sensor | Follows Category 1. **LOSO** for MoCap; the radar's is personal, as in Category 1. |
| MoCap exit factor | MoCap bout continues down to this × entry | **LOSO**, against the labels. Definition. |
| MoCap confirmation factor | a MoCap run counts only if it exceeds this × entry once | **LOSO**, against the labels. Definition. |
| `max_gap`, `min_bout` (`BoutRule`) | gap filling and minimum bout (now 0 s and 1 s) | **LOSO**, against the labels. Definition, and the same for both sensors. |
| radar confirmation ε | radar run must exceed entry + ε once | **LOSO**, on the bout count against MoCap. B and deployment: LOSO (too few bouts in a clip); it moves with the personal entry. C: also tested as a person parameter (person_1, person_25 count biases). |
| radar exit δ | radar bout continues down to entry − δ | **LOSO**, on the mean bout duration against MoCap. As ε. |
| fragmentation definition | non-bout → bout transition probability | **Fixed.** |
| `BOUT_MATCH_IOU` | bout matching for the evaluation (IoU ≥ 0.5) | **Fixed.** Evaluation only. |

### Category 3: postural transitions

| Parameter | Meaning | Calibration strategy |
|---|---|---|
| `hip_stand`, `trunk_lie` | MoCap posture: pelvis height for standing, trunk angle for lying | **LOSO**, against the script. Reference definition. |
| `low_floor`, MoCap `min_hold_s` | MoCap floor threshold, minimum posture hold | **LOSO**, on the transitions against the script. Definition. |
| `h_stand` | radar: corrected top-of-body height above which the person stands | **LOSO + person.** Stature. B: LOSO + *w* × shift; the shift is the mean of how far the person's clip standing and sitting heights lie from the other persons' clip heights. C: LOSO + *w*<sub>own</sub> × (threshold fitted against MoCap on the person's other recordings − LOSO); *w*<sub>own</sub> is calibrated on the training persons on mean posture F1 (it came out 0.25–0.5). Deployment: the session's standing step. |
| `h_sit` | radar: height above which the person sits (sitting vs lying) | **LOSO + room.** Chair and bed heights. B and C: LOSO + *w* × the mean shift of the person's clip sitting (chair and bed edge) and lying heights from the other persons' clips. *w* is one weight for the three thresholds, calibrated on mean posture F1 by simulating the clip on the training persons (it came out 0.25 in B and 0 in C for every person). Deployment: the same, from the session. |
| `h_floor` | radar: height below which the person is on the floor (lying vs floor) | **LOSO + room.** Bed height. B and C: LOSO + *w* × half the lying shift (the floor itself doesn't move); same *w* as `h_sit`. This doesn't fix the floor errors, which come from lying heights dipping below the threshold and from the floor rarely being seen; those need rule changes. With a low bed, lying in bed and sitting on the floor overlap in height, so the bed position is then needed (last row). Deployment: the same, from the session. |
| off-axis correction *k* | height gained per metre from the radar axis | **LOSO** (0.10–0.15 m/m). **Installation** if the mounting height differs from January; checked with the off-axis standing step. |
| radar `min_hold_s` | minimum posture hold for a radar transition | **LOSO**, on transition F1. Sensor mapping: it absorbs the radar's posture flicker, and equals the MoCap hold (3 s) by calibration, not by definition. Not personal: a clip or a person's recordings hold too few transitions (about 12 per recording). |
| peak Doppler quantile *q* | power-weighted quantile of the person's speeds (0.95) | **LOSO.** It is 0.95 in every fold, which shows a stable population optimum, not the absence of personal offsets: the mean error per person ranges from −0.50 to +0.02 m/s. A personal speed scale (as the Category 1 energy factor) is a candidate for setup C, once the radar's extra transitions are reduced. Not in the experiment yet. |
| `HEIGHT_SMOOTH_S`, `TOP_QUANTILE` | height smoothing (1 s median), top-of-body quantile (0.9) | **Fixed.** |
| burst classifier and its detection threshold | comparison method | **LOSO.** Not personalised (comparison only). |
| bed and chair positions | where the person rests | Not used yet. **Room**, from the first days' rest positions; candidates for location-dependent height thresholds, and needed for `h_floor` when the bed is low. |
| peak window `PEAK_HALF_S` | the peak Doppler is the maximum within this of a transition (1 s) | **Fixed.** |
| epoch posture | an epoch's posture: its most frequent frame posture (posture times) | **Fixed.** |
| `LABEL_MARGIN_S` (shared) | the scripted posture timeline is unknown within 2 s of a postural label | **Fixed.** |
| counted kinds | sit-to-stand, stand-to-sit, lie-to-sit (`COUNTED`) | **Fixed.** Definition. |
| `MATCH_TOL_S` | transitions match within ±1 s (SCOPE.md) | **Fixed.** Evaluation only. |
| `MIN_CLASS_BURSTS`, `NMS_S` | the burst classifier's minimum class size (10) and one event per direction per 1.5 s | **Fixed.** Comparison method only. |
| clip standing limit | the clip's standing epochs within 2 m of the axis | **Fixed.** Experiment design (see the clip above). |
