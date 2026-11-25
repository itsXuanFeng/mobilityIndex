# Category 3: postural transitions and posture

**What we measure.**

- **Transitions:** how often a person goes from sitting to standing, from standing to sitting, and from lying to sitting.
- **Peak speed of a transition:** how fast the person rises or sits down. Sit-to-stand speed is a known clinical marker of physical function, as in the five-times sit-to-stand test.
- **Time in each posture:** standing, sitting, lying, and on the floor.

## How the radar sees posture

The radar looks down from the ceiling, so it can measure **the height of the top of the body**: high when standing, lower when sitting, lowest when lying or on the floor. This needs the direction of each reflection as well as its distance ([angle.md](angle.md)); without the direction, a person far from the radar looks lower than they are.

- **Posture is followed over time**, by a small probabilistic model (a hidden semi-Markov model) rather than decided second by second. Two heights are its evidence: the top of the body, and the middle of the body, which tells lying in bed from lying on the floor. A posture changes only when the evidence over several seconds supports it, mostly while the person moves, and lasts at least as long as in the MoCap definition. Breathing and small movements keep a still person visible most of the time; when the radar loses the person, there is no evidence and the posture stays.
- **Transitions:** the changes of that posture, defined the same way as on MoCap.
- **Peak speed:** from the speeds of the person's own reflections during the transition.

Off the radar's axis, the radar sees less of the top of the head, so heights read low, by an amount that depends on the posture. The model learns this per posture from the data. Every posture change is allowed, also from a chair or bed straight to the floor, and the model learns nothing about the order of the scripted tasks.

The earlier version, three height thresholds per second, is kept as a comparison.

## How we check it

- **MoCap posture** comes from the skeleton (pelvis height, knee height, trunk angle). It is calibrated against the scripted postures, leaving the person out, and reproduces almost all of the scripted changes.
- **Radar against MoCap:** posture second by second, transitions matched one to one within a second, and the time in each posture and the transition counts per recording.
- All radar settings are calibrated leaving the evaluated person out. The posture model was also checked leaving out a whole furniture layout, and the person and the layout together.

## Where it stands

- Standing, sitting, lying and the floor are recognised well second by second, and the time spent in each posture closely follows MoCap. This is the first Category 3 metric where the radar does better than the script predicts.
- Nearly all transitions are found, with the right kind and timing, also far off the radar's axis, and the model works as well in a furniture layout it has not seen.
- The peak speed agrees with MoCap's, once a leakage effect of strong reflections in the radar's speed measurement is avoided.
- About an hour of MoCap recordings is enough to calibrate the model.

**Open challenges:**
- **The script fixes the counts:** every recording has the same number of scripted transitions. Validating the counts needs free-living recordings.
- **Short postures:** in the scripted tasks, nearly every posture shorter than a few seconds is a passing phase of one movement (kneeling while getting up, bending). How short a real posture can be, and whether the radar catches it, needs free-living recordings.
- **Another room:** all recordings share one room and one radar position; a different ceiling or mounting is untested.
- **Next steps:** the direction of the Doppler as evidence of going down or up, and the duration of a transition as an extra metric.
