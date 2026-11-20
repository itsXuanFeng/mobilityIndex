# Category 1: total movement volume

**What we measure.** How much a person moves over a period of time:

- **active time:** the time spent in whole-body movement, such as walking or changing position;
- **inactive time:** the rest of the time, such as sitting, lying or standing still (arm movements like eating or drinking count as inactive);
- **total activity energy:** how vigorous the movement is, summed over the period.

These are basic building blocks of a mobility profile: a decline in daily movement is one of the earliest signs of frailty.

## How the radar sees movement

The radar hangs on the ceiling and looks down. Every 50 ms it measures, for each distance from the radar, how much is moving at each speed towards or away from it. Furniture and walls don't move, so they are removed; what remains is the person moving.

- Each second, the radar decides **active** or **inactive** from how many distance–speed cells light up above their usual background level.
- For the **energy**, the radar also measures the *direction* of each reflection ([angle.md](angle.md)). It can then keep only the reflections that belong to the person, and sum the squared speeds of their body parts, like a kinetic-energy measure.

The current version estimates each cell's background level over the whole recording. A version that uses only past data, and so can run online in a home, has been prototyped and performs as well; it is not yet part of the pipeline.

## How we check it

The radar is compared with marker-free **motion capture (MoCap)** of the full body, recorded at the same time in a lab apartment. Participants followed a script of everyday activities: walking, kitchen tasks, eating at a chair, lying in bed, getting down to the floor and back up, a caregiver visit.

- **"Active" is defined on MoCap** as whole-body movement, with a speed threshold calibrated against the activity labels of the script.
- **The radar is calibrated against MoCap** without the person being evaluated (leave-one-person-out), so the results show how it works for a new person.
- **Comparison:** second by second (does the radar call the same seconds active?), and per recording (are the totals the same?).

## Where it stands

- Active and inactive time from the radar agree closely with MoCap, second by second and per recording.
- The radar's activity energy ranks recordings in the same order as MoCap's. Once converted to MoCap units, it is clearly closer than a guess without a sensor.
- An early finding: radar and MoCap must be aligned in time precisely. Earlier work underestimated the radar because of a misalignment.

**Open challenges:** a second person in the room (a caregiver) is still mixed with the resident; the signal is weaker far from the radar; and reading the raw radar data is slow.
