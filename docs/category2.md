# Category 2: activity bout structure

**What we measure.** Not only *how much* a person moves (Category 1), but *how* the movement is spread over time:

- **number of bouts:** how many separate periods of activity;
- **mean and variance of bout duration:** how long these periods last, and how much that varies;
- **fragmentation index:** how often rest is interrupted by activity.

In studies with wearables, shorter and more fragmented activity is linked to fatigability, falls and functional decline in older adults. The radar could measure the same thing without anything to wear.

## How bouts are defined

Category 2 builds on the second-by-second activity level of [Category 1](category1.md) and on the changes of posture of [Category 3](category3.md). The same rule applies to the radar and to MoCap, each on its own activity level and its own changes of posture:

1. **Start:** a bout starts when the activity level crosses an active threshold.
2. **End:** it ends only when the level drops clearly below that threshold, so a brief pause doesn't split one movement in two.
3. **Confirmation:** it only counts if the level is clearly above the threshold at least once, so small movements that barely cross it don't become bouts. A change of posture (sitting up, standing up, lying down) also confirms it: that is a whole-body movement even when it is slow or under a blanket.
4. **Clean-up:** very short gaps can be filled and very short bouts dropped. The calibration currently finds this unnecessary: the confirmation already removes such blips.

**The radar's level is corrected for distance.** The farther the person is from the radar, the less of the same movement it sees. The radar estimates where the person is and corrects its activity level for that distance, so that a movement counts the same anywhere in the room.

None of these thresholds is set by hand. On MoCap they are calibrated against the activity labels of the script. On the radar, the distance correction and the active threshold are calibrated against MoCap, and the end and the confirmation use one margin, the same distance below and above the active threshold. Everything is calibrated leaving the evaluated person out, and comes out the same for nearly every person.

## How we check it

- **Bout by bout:** each radar bout is matched to a MoCap bout by overlap, and we check the start and end times.
- **Per recording:** the four metrics from the radar and from MoCap, compared with each other and with a guess that uses no sensor at all.
- **Against the script:** the bouts of both sensors are also compared with the activity labels of the script, a reference that doesn't depend on MoCap.
- **In bed and elsewhere:** the bouts are also compared separately in bed and out of bed, with the radar's own posture.
- **With Category 3:** every change of posture should fall inside a bout of the same sensor.

## Where it stands

- Radar and MoCap find largely the same bouts, one by one, with matching start and end times.
- All four metrics follow MoCap clearly better than the no-sensor guess. The radar's bouts are slightly shorter than MoCap's on average.
- Against the script's activity labels, the radar's bouts agree almost as well as MoCap's do.
- Out of bed, radar and MoCap agree closely. In bed, the radar counts small settling movements as bouts that MoCap doesn't.
- Nearly every change of posture lies inside a bout, for both sensors.

**Open challenges:**
- **The scripted protocol:** every recording follows the same script, so the number of bouts hardly varies between people. The data confirm the radar sees the right bouts. They can't yet show that bout counts tell people apart; that needs unscripted, free-living recordings.
- **Movement in bed:** the two sensors split movement under a blanket into bouts differently, and MoCap itself is least reliable there.
- **Online use:** the changes of posture come from a model that currently looks at a whole recording; running it live needs a short delay.
