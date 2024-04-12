
import cupy  # NOQA


def parks_mclellan_bp(n, freqs, amplitudes, weights, eps=0.01, nmax=4):
    pass


def parks_mclellan(n, freqs, amplitudes, weights,
                   type='bandpass', eps=0.01, nmax=4):
    if type == 'bandpass':
        return parks_mclellan_bp(n, freqs, amplitudes, weights, eps, nmax)
