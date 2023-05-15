
import cupy
from cupyx.scipy.signal._spectral_impl import _lombscargle


def lombscargle(x, y, freqs, precenter=False, normalize=False):
    """
    lombscargle(x, y, freqs)

    Computes the Lomb-Scargle periodogram.

    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
    extended by Scargle [2]_ to find, and test the significance of weak
    periodic signals with uneven temporal sampling.

    When *normalize* is False (default) the computed periodogram
    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
    signal with amplitude A for sufficiently large N.

    When *normalize* is True the computed periodogram is normalized by
    the residuals of the data around a constant reference model (at zero).

    Input arrays should be one-dimensional and will be cast to float64.

    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values.
    freqs : array_like
        Angular frequencies for output periodogram.
    precenter : bool, optional
        Pre-center amplitudes by subtracting the mean.
    normalize : bool, optional
        Compute normalized periodogram.

    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.

    Notes
    -----
    This subroutine calculates the periodogram using a slightly
    modified algorithm due to Townsend [3]_ which allows the
    periodogram to be calculated using only a single pass through
    the input arrays for each frequency.
    The algorithm running time scales roughly as O(x * freqs) or O(N^2)
    for a large number of samples and frequencies.

    References
    ----------
    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -
           Statistical aspects of spectral analysis of unevenly spaced data",
           The Astrophysical Journal, vol 263, pp. 835-853, 1982
    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
           periodogram using graphics processing units.", The Astrophysical
           Journal Supplement Series, vol 191, pp. 247-253, 2010

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    welch: Power spectral density by Welch's method
    spectrogram: Spectrogram by Welch's method
    csd: Cross spectral density by Welch's method
    """

    x = cupy.asarray(x, dtype=cupy.float64)
    y = cupy.asarray(y, dtype=cupy.float64)
    freqs = cupy.asarray(freqs, dtype=cupy.float64)
    pgram = cupy.empty(freqs.shape[0], dtype=cupy.float64)

    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1

    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    y_dot = cupy.zeros(1, dtype=cupy.float64)
    if normalize:
        cupy.dot(y, y, out=y_dot)

    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y

    _lombscargle(x, y_in, freqs, pgram, y_dot)

    return pgram
