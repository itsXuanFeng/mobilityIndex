# Category 3: postural transitions and posture

**What we measure.**

- **Transitions:** how often a person goes from sitting to standing, from standing to sitting, and from lying to sitting.
- **Peak speed of a transition:** how fast the person rises or sits down. Sit-to-stand speed is a known clinical marker of physical function, as in the five-times sit-to-stand test.
- **Time in each posture:** standing, sitting, lying, and on the floor.

## How the radar sees posture

The radar looks down from the ceiling, so it can measure **the height of the top of the body**: high when standing, lower when sitting, lowest when lying or on the floor. This needs the direction of each reflection as well as its distance ([angle.md](angle.md)); without the direction, a person far from the radar looks lower than they are.

- **Posture**, each second: from the measured height, with three height thresholds. Breathing and small movements keep a still person visible most of the time. When the radar loses the person, the last posture is kept.
- **Transitions:** a change in posture that lasts, by the same rule as on MoCap.
- **Peak speed:** from the speeds of the person's own reflections during the transition.

Off the radar's axis, the radar sees less of the top of the head, so heights read slightly low. A correction for the distance from the axis is calibrated from the data.

## How we check it

- **MoCap posture** comes from the skeleton (pelvis height, knee height, trunk angle). It is calibrated against the scripted postures, leaving the person out, and reproduces almost all of the scripted changes.
- **Radar against MoCap:** posture second by second, transitions matched one to one within a second, and the time in each posture and the transition counts per recording.
- All radar thresholds are calibrated leaving the evaluated person out.

## Where it stands

- Standing, sitting and lying are recognised well second by second, and the time spent in each posture closely follows MoCap. This is the first Category 3 metric where the radar does better than the script predicts.
- Most transitions are found, with the right kind and timing.
- The peak speed agrees with MoCap's, once a leakage effect of strong reflections in the radar's speed measurement is avoided.

**Open challenges:**
- **The script fixes the counts:** every recording has the same number of scripted transitions. Validating the counts needs free-living recordings.
- Transitions far off the radar's axis are found less often.
- A person lying still on the floor is hard to see, and lying in bed is close in height to the floor.
- **Next steps:** a probabilistic posture model, and the duration of a transition as an extra metric.
