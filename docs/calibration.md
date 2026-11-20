# Calibration: population first, then personal

Every method in Categories 1–3 has a few parameters, such as thresholds on the activity level or on the measured height. This page explains how they are set.

## Population calibration (the default)

Every parameter is calibrated on **other people**, never on the person being evaluated (leave-one-person-out). The results therefore show how the radar works for a **new resident with no setup at all**. That is the main number for how well the methods generalise.

Nothing is tuned by hand. The reference itself (what counts as "active" on MoCap, what a bout is, what a posture is) is calibrated against the labels of the scripted activities, also leaving the person out.

## Personal calibration (being tested)

Some remaining errors are steady for a given person, not random: the radar consistently reads one person's energy a little low, or another person's heights depend on their stature and furniture. A short calibration in the resident's own room could remove such offsets.

**Principles:**

- **Definitions stay population-wide.** What counts as active, what a bout is and what each posture means are never personalised. Otherwise a slow resident's thresholds would drift down, and the calibration would hide exactly the decline we want to measure.
- **Only the sensor is personalised:** the body (stature, size), where the person is relative to the radar, and the room (bed and chair heights).
- **Adjust, don't replace.** A personal parameter starts from its population value and moves only as far as the person's own data supports.

**In deployment**, the personal data would come from a short guided session at installation: a few minutes of standing, walking, sitting on their chair, and lying in bed, with a caregiver marking each step. No motion capture is needed.

**The experiment.** On the January data, three setups are compared on the same data:

1. population calibration only;
2. population + a short labelled clip of the recording, which simulates the installation session;
3. population + the person's other recordings + the clip, an upper bound with far more personal data than a session would give.

**First results:** the activity threshold gains little from personalisation, while the energy scale does gain from more personal data. The session design is being adjusted so it can capture what matters, for example with a timed walk.
