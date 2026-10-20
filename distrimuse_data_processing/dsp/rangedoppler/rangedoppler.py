import numpy as np

RANGE_FFT = 256
DOPPLER_FFT = 128
RANGE_BINS = RANGE_FFT // 2


def range_doppler(data, n_range_fft=RANGE_FFT, n_doppler_fft=DOPPLER_FFT):
    """
    Magnitude range-Doppler map of one antenna pair.
    :param data: (chirps, samples) ADC samples
    :return: (n_range_fft // 2, n_doppler_fft), Doppler axis fftshifted (zero velocity in the middle)
    """
    data = data.T[1:]  # (samples, chirps), first sample of every chirp dropped
    n_samples, n_chirps = data.shape

    # range FFT: Hann window, zero padding, positive range bins only
    window = np.hanning(n_samples)
    x = np.zeros((n_range_fft, n_chirps), dtype=complex)
    start = int((n_range_fft - n_samples) / 2)
    x[start:start + n_samples, :] = data * window[:, None]
    X = np.fft.fft(x, n_range_fft, 0) / np.sum(window) * (2.0 / 2048)
    X = X[:n_range_fft // 2, :]

    # clutter removal: static reflections are the mean over the chirps of each range bin
    X = X - np.mean(X, axis=1)[:, None]

    # Doppler FFT: Hann window, zero padding
    window = np.hanning(n_chirps)
    rd = np.zeros((X.shape[0], n_doppler_fft), dtype=complex)
    start = int((n_doppler_fft - n_chirps) / 2)
    rd[:, start:start + n_chirps] = X * window
    rd = np.fft.fft(rd, n_doppler_fft, 1) / np.sum(window)
    return np.abs(np.fft.fftshift(rd, axes=1))


def range_doppler_db(frame):
    """
    Range-Doppler map in dB of TX0/RX0.
    :param frame: (loops, chirps, rx, samples)
    :return: (RANGE_BINS, DOPPLER_FFT); -inf where the magnitude is 0 (all-zero frame)
    """
    with np.errstate(divide='ignore'):
        return 20 * np.log10(range_doppler(frame[:, 0, 0, :]))


def is_null(doppler):
    """frame without data: its time-Doppler row is all 0, or all -inf (all-zero samples)"""
    return bool(np.all(doppler == 0) or np.all(np.isneginf(doppler)))
