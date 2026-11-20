# Category 2: activity bout structure

**What we measure.** Not only *how much* a person moves (Category 1), but *how* the movement is spread over time:

- **number of bouts:** how many separate periods of activity;
- **mean and variance of bout duration:** how long these periods last, and how much that varies;
- **fragmentation index:** how often rest is interrupted by activity.

In studies with wearables, shorter and more fragmented activity is linked to fatigability, falls and functional decline in older adults. The radar could measure the same thing without anything to wear.

## How bouts are defined

Category 2 builds on the second-by-second activity level of [Category 1](category1.md). The same rule applies to the radar and to MoCap, each on its own activity level:

1. **Start:** a bout starts when the activity level crosses the active threshold.
2. **End:** it ends only when the level drops clearly below that threshold, so a brief pause doesn't split one movement in two.
3. **Confirmation:** it only counts if the level is clearly above the threshold at least once, so small movements that barely cross it don't become bouts.
4. **Clean-up:** very short gaps are filled and very short bouts dropped.

None of these thresholds is set by hand. On MoCap they are calibrated against the activity labels of the script. On the radar they are calibrated against MoCap, leaving the evaluated person out.

## How we check it

- **Bout by bout:** each radar bout is matched to a MoCap bout by overlap, and we check the start and end times.
- **Per recording:** the four metrics from the radar and from MoCap, compared with each other and with a guess that uses no sensor at all.

## Where it stands

- Radar and MoCap find largely the same bouts, one by one, with matching start and end times.
- All four metrics follow MoCap without a systematic bias, and are closer to it than the no-sensor guess.

**Open challenges:**
- **The scripted protocol:** every recording follows the same script, so the number of bouts hardly varies between people. The data confirm the radar sees the right bouts. They can't yet show that bout counts tell people apart; that needs unscripted, free-living recordings.
- **Movement in bed:** the two sensors split movement under a blanket into bouts differently, and MoCap itself is least reliable there.
