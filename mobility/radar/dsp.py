"""
Range-Doppler power of every virtual channel, batched on the GPU. Same steps as the original preprocessing code
(commit dc5c08c, rangedoppler.py): drop ADC sample 0, Hann, range FFT, keep 128 bins, per-frame clutter removal (mean
over the chirps), Hann, zero-pad 100 -> 128 chirps, Doppler FFT, fftshift.
"""
import torch

from ..config import N_DOPPLER, N_RANGE

RANGE_FFT = 256


class RangeDoppler:
    def __init__(self, device, n_samples=256, n_loops=100):
        self.device = device
        self.w_range = torch.hann_window(n_samples - 1, periodic=False, device=device)
        self.w_doppler = torch.hann_window(n_loops, periodic=False, device=device)
        self.scale_range = 1 / self.w_range.sum() * 2 / 2048
        self.scale_doppler = 1 / self.w_doppler.sum()
        self.pad = (N_DOPPLER - n_loops) // 2

    def power(self, frames):
        """
        :param frames: (n, loops, tx, rx, samples) complex64, numpy or torch
        :return: (n, tx, rx, N_RANGE, N_DOPPLER) power (linear), Doppler fftshifted (zero velocity at N_DOPPLER // 2)
        """
        x = torch.as_tensor(frames, device=self.device)[..., 1:] * self.w_range
        r = torch.fft.fft(x, RANGE_FFT, dim=-1)[..., :N_RANGE] * self.scale_range     # (n, loops, tx, rx, range)
        r = r - r.mean(dim=1, keepdim=True)                                            # static reflections
        r = r * self.w_doppler[None, :, None, None, None]
        r = torch.nn.functional.pad(r.permute(0, 2, 3, 4, 1), (self.pad, self.pad))   # (n, tx, rx, range, loops)
        d = torch.fft.fftshift(torch.fft.fft(r, dim=-1), dim=-1) * self.scale_doppler
        return d.real ** 2 + d.imag ** 2
