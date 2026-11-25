# Locating the person: the angle of arrival

**Why it is needed.** Distance and speed alone don't tell where a person is: a reflection 2 m from the ceiling radar could be a person standing far away or sitting right below it. The radar used here has a small 2-D grid of receive channels. From the tiny delays between them it can estimate the **direction** of each reflection, and with the distance, its **3-D position**.

**What it gives:**

- the **height of the person's body** (its top and its middle), from which posture is followed ([Category 3](category3.md));
- **which reflections belong to the person**, so walls, furniture or other moving objects don't count towards their activity energy ([Category 1](category1.md));
- the person's **position in the room**, for example near the bed or the chair.

## How it is done

The raw radar data are read once more (the first pass, for Categories 1–2, averages the channels and so loses the direction). For each moment, the cells that hold something moving or breathing are kept, with the signal of every channel. Only these measurements are stored. The direction finding, the corrections for the antenna layout and the radar's orientation in the room all happen afterwards, so they can be improved without reading the raw data again.

## How we check it

The radar's 3-D positions are compared with the joints of the MoCap skeleton: the nearest moving reflection against the nearest joint, in one recording per furniture layout.

## Where it stands

- The positions agree with MoCap to within the size of a body part, horizontally.
- The height from the direction is much more accurate than the height from the distance alone, especially away from the radar's axis.
- The radar's orientation in the room is simple and the same in every recording.
- The whole January campaign has been processed.

**Open challenges:** separating two people (resident and caregiver), which would make about a third more of the radar time usable; and reading the raw data, which is limited by the network speed.
