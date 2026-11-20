1. Check and protect generalisation (do first)

- Leave one layout out, not just one person. Each layout has a different chair and bed position, and the off-axis term is learned per posture. Holding out each layout in turn shows whether the 0.96 holds in a room the model has never seen, which matters more for deployment than another F1 point. It's cheap: the same code with a different fold key.
- Replace the 4 s hold with a duration model. A hidden semi-Markov model learns how long a posture lasts instead of a hard minimum. The 4 s hold would erase short real postures in free living, such as briefly sitting on the bed edge before standing.
- Don't let the transition probabilities learn the script. They are uniform over allowed paths now, which is good, and they should stay generic. The "all transitions allowed" run that's still going tells us how much the path constraint is doing.

2. Make the model better

- Use the Doppler direction as evidence for a transition. The activity gate is "on" 90% of the time, so it adds almost nothing now. A stricter rule would be that posture can only go down while the person's points move away from the radar (positive signed Doppler), and only up while they move toward it. That would remove flips without real vertical motion, like those at the bed, and should sharpen timing (median error is now 0.26 s). The signed velocity is already computed (v_signed in the scratch features).
- Report confidence. Forward-backward gives a probability per posture per epoch. Posture times could become expected times (the sum of those probabilities), with low-confidence stretches flagged instead of forced into one posture.
- Learn the furniture per home (for later, free living). Over days, where and at what height a person sits or lies can be learned without labels and fed back as a per-home prior. That adapts to the room, not the person, so it avoids the over-personalisation you were worried about.

3. Better Category 3 metrics for a mobility index

- Transition duration. Sit-to-stand time is a known clinical marker (as in the 5× sit-to-stand test), and the HMM gives each transition a start and end. It is measurable against MoCap now, and unlike the counts the script doesn't fix it.
- Vertical speed of the top of the body during a transition, as a peak-speed measure next to peak Doppler. It doesn't depend on the angle to the radar or on how much power the fastest body part returns. Peak Doppler under-reads the floor movements by 0.3–0.5 m/s for that reason.
- Failed attempts: a rise that starts and falls back to sitting, as an extra HMM state. This campaign has no such events to validate it, so it is a design note for later recordings.

4. Inherited limits

- Separating the person from a caregiver with the angle (tracking more than one person) would make about a third more radar time usable, for all three categories.
- Validating the counts still needs recordings that aren't scripted; no model change fixes that.