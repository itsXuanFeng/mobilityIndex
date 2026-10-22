"""Category 1: total movement volume. See docs/category1.md."""
import numpy as np

from . import metric


@metric('total_activity_energy', category=1, unit='energy x s (source units)')
def total_activity_energy(s):
    """movement energy integrated over the valid epochs. Radar: distance-corrected moving power (p_r4); MoCap: mean
    squared joint speed. Different units: compare across recordings (rank / ICC on standardised values)."""
    return float(np.nansum(s.energy[s.valid]) * s.epoch_s)


@metric('active_time', category=1, unit='min')
def active_time(s):
    """time in active epochs"""
    return float((s.active & s.valid).sum() * s.epoch_s / 60)


@metric('inactive_time', category=1, unit='min')
def inactive_time(s):
    """time in inactive epochs (of the valid ones)"""
    return float((~s.active & s.valid).sum() * s.epoch_s / 60)
