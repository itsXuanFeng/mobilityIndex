# FMCW preprocessing: raw radar data → time-Doppler maps

This pipeline turns each raw FMCW radar recording into a **time-Doppler map**: one row per radar frame (20 frames/s), one column per Doppler bin. Optionally, it stores the per-frame **range-Doppler maps** instead.

## Output

tested with January and August data

`preprocessing/output/<campaign>/<person>/<scenario>/time_doppler_max.npy`

| | |
|---|---|
| Shape | (T, 128): T frames × 128 Doppler bins |
| dtype | float64 |
| Rows | **row *j* = frame *j***, recorded *j* / 20 s after the first frame of the recording. This holds for every row, including after missing data (see "NaN rows"). |
| Columns | Doppler bins, fftshifted: **index 64 is zero Doppler**. Static reflections are removed per frame, so the zero-Doppler bin is suppressed. |
| Values | Relative dB (20·log10 of the FFT magnitude), not calibrated. About −75 dB without motion; 1st–99th percentile −78 to −47 dB (August campaign). |
| NaN | Whole rows of NaN for frames without data; see "NaN rows". There are no partly-NaN rows and no ±inf. |

Example:

```python
import numpy as np
td = np.load('preprocessing/output/August_2025/person_1/scenario_3/time_doppler_max.npy')
t = np.arange(len(td)) / 20          # time of each row, in seconds
no_data = np.isnan(td).all(axis=1)   # frames without data
```

### Optional: range-Doppler maps

With `--output range_doppler`, the pipeline writes `range_doppler.npy` instead of the time-Doppler map:
- shape (T, 128 range bins, 128 Doppler bins);
- the same rows (row *j* = frame *j*), with NaN frames for frames without data;
- the time-Doppler map is its maximum over range, `range_doppler.max(axis=1)`;
- dtype float32 by default (`--rd_dtype float16|float32|float64`). float32 takes ~0.8 GB per 10 min of recording.

## Input: the raw recordings

| Campaign | Raw data |
|---|---|
| January_2026 | `/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/January_2026/Measurement Campaign Januari 2026/FMCW/person_<N>/scenario_<M>/` |
| August_2025 | `/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/August_2025/FMCW/person_<N>/scenario_<M>/` |

Each recording folder holds the raw ADC data captured by a TI DCA1000 board, `test2x_1_Raw_0.bin`, `test2x_1_Raw_1.bin`, …, and the capture logs (`test2x_1_LogFile.txt`, `test2x_1_Raw_LogFile.csv`).

Radar settings (identical in both campaigns; read from `dsp/rangedoppler/config.mmwave.json`):

| Setting | Value |
|---|---|
| Antennas | 3 TX (time-division: chirp 0/1/2 = TX 0/1/2), 4 RX |
| Chirp | 256 complex ADC samples |
| Frame | 100 loops × 3 chirps (one per TX) |
| Frame period | 50 ms → 20 frames/s |
| Bytes per frame | 100 × 3 × 4 × 256 samples × 4 bytes = 1,228,800 |

**Byte layout.** A recording is one byte stream: frame → loop → chirp (TX) → RX → samples, with every 8 bytes holding two complex samples as little-endian int16 `I0, I1, Q0, Q1` (samples `I0 + jQ0` and `I1 + jQ1`). Frame *i* is stream bytes [*i* · 1,228,800, (*i* + 1) · 1,228,800).

**Files.** The DCA1000 splits the stream into `.bin` files of 1,073,741,760 bytes (873.8 frames) each, regardless of frame boundaries, so frames can straddle two files. `Raw_<k>.bin` holds stream bytes from *k* · 1,073,741,760 onwards; the last file is shorter.

## Pipeline

Per recording:

1. **Locate the files** (`dsp/recordings.py`). The `.bin` files are sorted by their number *k*, and each is placed at stream byte *k* · 1,073,741,760. A file listed in `EXCLUDE` is left out.
2. **Read frames** (`dsp/raw_data/reader.py`). The files are read as one byte stream, cut into frames of 1,228,800 bytes, and decoded to complex64 arrays of shape (100 loops, 3 chirps, 4 RX, 256 samples). A frame that straddles two files is assembled from both. Frames that are not completely on disk are skipped (they become NaN rows). An incomplete frame at the very end of the recording is dropped.
3. **Range-Doppler map** (`dsp/rangedoppler/rangedoppler.py`), from one antenna pair, TX 0 / RX 0, i.e. a (100 chirps × 256 samples) block:
   1. drop the first ADC sample of every chirp → 255 samples;
   2. Hann window over the samples, zero-pad to 256, **range FFT**, scale by 1 / (sum of the window) · 2 / 2048, keep the first 128 range bins;
   3. **clutter removal**: subtract, per range bin, the mean over the 100 chirps (removes static reflections);
   4. Hann window over the chirps, zero-pad 100 → 128 (14 zeros on each side), **Doppler FFT**, scale by 1 / (sum of the window), `fftshift`, magnitude;
   5. 20 · log10 → a (128 range × 128 Doppler) map in dB.
4. **Time-Doppler row**: the maximum over the 128 range bins → 128 values.
5. **Null frames**: a frame whose row is −inf everywhere (all its samples are zero) or 0 everywhere has no data and becomes a NaN row.
6. **Save**: an array of T rows, where T is the number of frames up to the last complete frame on disk. Rows without data are NaN.

## NaN rows

A **NaN row** (all 128 values NaN) marks a frame without data. These rows are kept in place, so row *j* is always frame *j* and the time axis never shifts. There are two causes:

- **Data missing from the files**: a `.bin` file is truncated or absent. Every frame that is not completely on disk is NaN, including the frames cut at the edges of the missing part.
- **All-zero frames**: frames whose samples are all zero. These come from packets lost during the capture, which the DCA1000 fills with zeros, or from zero regions in a damaged file.

Recordings with NaN rows:

| Recording | Rows | NaN rows | Cause |
|---|---|---|---|
| January_2026 person_23/scenario_5 | 9,257 | 858–873 | `Raw_0` truncated |
| January_2026 person_101/scenario_100 | 31,880 | 0–26,214 | `Raw_0`–`Raw_29` absent (data starts at `Raw_30`) |
| August_2025 person_7/scenario_5 | 6,258 | 1747–2621, 4369–5242 | `Raw_2`, `Raw_5` absent |
| August_2025 person_7/scenario_6 | 5,940 | 873–1747 | `Raw_1` absent |
| August_2025 person_19/scenario_6 | 7,387 | 3453–3495, 5090–5242, 5965–6116; 5947–5964 | `Raw_3`, `Raw_5`, `Raw_6` truncated; all-zero frames |
| August_2025 person_19/scenario_90 | 1,199 | 654–873 | `Raw_0` truncated |
| August_2025 person_8/scenario_1 | 6,974 | 5752–5905 | all-zero frames (packets lost during the capture) |

All other August recordings have no NaN rows. For January, the table lists the missing files. All-zero frames, if any, are reported in the log when the campaign is processed (`… NaN (… missing, … all-zero)`).

**Handling NaN.** Code that reads these maps must expect NaN rows where a window or event overlaps one. For example:
- skip or mask such windows: `np.isnan(td).all(axis=1)` gives the frames without data;
- use NaN-aware statistics for normalization (`np.nanmean`, `np.nanstd`);
- note that averaging adjacent Doppler bins keeps a NaN row NaN.

Frames in which only a few packets were lost and zero-filled are not marked: they keep finite, slightly distorted values.

## Running the pipeline

From the repository root:

```
python preprocessing/dsp/time_doppler.py --campaign August_2025 --workers 16
python preprocessing/dsp/time_doppler.py --campaign January_2026 --only person_1/scenario_1 person_2/scenario_3
python preprocessing/dsp/time_doppler.py --only person_1/scenario_1 --output range_doppler
```

| Option | Default | |
|---|---|---|
| `--campaign` | `January_2026` | `January_2026` or `August_2025` |
| `--only PERSON/SCENARIO …` | all recordings | process only these recordings |
| `--workers` | 1 | recordings processed in parallel |
| `--overwrite` | off | recompute recordings that already have an output file (otherwise they are skipped) |
| `--output` | `time_doppler` | `time_doppler` or `range_doppler` |
| `--rd_dtype` | `float32` | dtype of `range_doppler.npy` |

- **Output files.** Each output is first written to `<name>.tmp.npy` and renamed when complete, so an interrupted run never leaves a partial output. Temporary files of failed or interrupted recordings are removed, also on Ctrl-C.
- **Failures.** A recording that fails doesn't stop the others. The traceback is printed, and the run ends with a list of the failed recordings and exit code 1.
- **Log.** Each recording prints one summary line, e.g. `person_19/scenario_6: time_doppler_max.npy done, 7387 frames, 366 NaN (348 missing, 18 all-zero), 423 s`. Every gap in the data gets one line, e.g. `…/test2x_1_Raw_4.bin: gap in the data, frames 3453-3495 missing or incomplete`.
- **Run time.** Reading the raw data from the network drive is the bottleneck: the August campaign (0.9 TB, 120 recordings) took about 2 h 20 min with 16 workers.

## Code

| File | |
|---|---|
| `dsp/time_doppler.py` | command-line entry point; per recording: frames → time-Doppler map (and range-Doppler maps), writing the output, parallel runs |
| `dsp/recordings.py` | campaign paths, file positions (`FILE_BYTES`), excluded files, finding recordings, counting the frames on disk |
| `dsp/raw_data/reader.py` | radar configuration and decoding of the `.bin` byte stream into frames |
| `dsp/raw_data/rf_config.py` | parser of the mmWave Studio configuration |
| `dsp/rangedoppler/rangedoppler.py` | range-Doppler map of a frame, null-frame test |
| `dsp/rangedoppler/config.mmwave.json` | radar configuration used for all recordings |
