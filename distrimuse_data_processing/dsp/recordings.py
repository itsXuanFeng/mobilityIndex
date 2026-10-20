"""Where the raw FMCW recordings are, and how each recording is split into .bin files."""

import glob
import os
import re

RAW_ROOTS = {
    'January_2026': '/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/January_2026/Measurement Campaign Januari 2026/FMCW',
    'August_2025': '/cw/dtaidata/dtaigeel/2021-NextPerception/DistriMuse/August_2025/FMCW',
}

# The DCA1000 splits a recording into .bin files of FILE_BYTES each (1024 MB rounded down to whole 1456-byte
# packets). Raw_<k> starts at byte k * FILE_BYTES of the recording, also when earlier files are truncated or missing.
FILE_BYTES = 1073741760

# .bin files left out: Raw_15 doesn't align with its position in the recording (Raw_12-14 are missing and Raw_11 is
# truncated), so its ~4 frames can't be placed
EXCLUDE = {
    ('January_2026', 'person_6', 'scenario_3'): ['test2x_1_Raw_15.bin'],
}


def natural_key(s):
    """sort key that orders numbers numerically: 'person_2' < 'person_10'"""
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def find_scenarios(campaign):
    """(person, scenario) of every recording with .bin files"""
    bin_files = glob.glob(f'{RAW_ROOTS[campaign]}/person_*/scenario_*/*_Raw_*.bin')
    scenarios = {tuple(f.split('/')[-3:-1]) for f in bin_files}
    return sorted(scenarios, key=lambda ps: natural_key('/'.join(ps)))


def recording_files(campaign, person, scenario):
    """the recording's .bin files in order, and the byte offset of each in the recording"""
    folder = os.path.join(RAW_ROOTS[campaign], person, scenario)
    excluded = EXCLUDE.get((campaign, person, scenario), [])
    numbered = {}
    for name in os.listdir(folder):
        match = re.search(r'_Raw_(\d+)\.bin$', name)
        if match and name not in excluded:
            numbered[int(match.group(1))] = os.path.join(folder, name)
    paths = [numbered[k] for k in sorted(numbered)]
    for path in paths:
        assert os.path.getsize(path) <= FILE_BYTES, f'{path} is larger than {FILE_BYTES} bytes'
    return paths, [k * FILE_BYTES for k in sorted(numbered)]


def frames_on_disk(paths, offsets, frame_size):
    """number of frames of the recording, up to the last complete frame on disk: the last frame whose bytes all lie in
    one contiguous run of files"""
    spans = []  # [start, end) byte ranges present, contiguous files merged
    for path, offset in zip(paths, offsets):
        end = offset + os.path.getsize(path)
        if spans and spans[-1][1] == offset:
            spans[-1][1] = end
        else:
            spans.append([offset, end])
    for start, end in reversed(spans):
        if end // frame_size > -(-start // frame_size):  # the span holds at least one whole frame
            return end // frame_size
    return 0
