"""Functions for FIR filter design."""

from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx

import cupy

__all__ = ["firls"]


# Scipy <= 1.12 has a deprecated `nyq` argument (nyq = fs/2).
# Remove it here, to be forward-looking.
def firls(numtaps, bands, desired, weight=None, fs=2):
    """
    FIR filter design using least-squares error minimization.

    Calculate the filter coefficients for the linear-phase finite
    impulse response (FIR) filter which has the best approximation
    to the desired frequency response described by `bands` and
    `desired` in the least squares sense (i.e., the integral of the
    weighted mean-squared error within the specified bands is
    minimized).

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter. `numtaps` must be odd.
    bands : array_like
        A monotonic nondecreasing sequence containing the band edges in
        Hz. All elements must be non-negative and less than or equal to
        the Nyquist frequency given by `fs`/2. The bands are specified as
        frequency pairs, thus, if using a 1D array, its length must be
        even, e.g., `cupy.array([0, 1, 2, 3, 4, 5])`. Alternatively, the
        bands can be specified as an nx2 sized 2D array, where n is the
        number of bands, e.g, `cupy.array([[0, 1], [2, 3], [4, 5]])`.
        All elements of `bands` must be monotonically nondecreasing, have
        width > 0, and must not overlap. (This is not checked by the routine).
    desired : array_like
        A sequence the same size as `bands` containing the desired gain
        at the start and end point of each band.
        All elements must be non-negative (this is not checked by the routine).
    weight : array_like, optional
        A relative weighting to give to each band region when solving
        the least squares problem. `weight` has to be half the size of
        `bands`.
        All elements must be non-negative (this is not checked by the routine).
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `bands`
        must be between 0 and ``fs/2`` (inclusive). Default is 2.

    Returns
    -------
    coeffs : ndarray
        Coefficients of the optimal (in a least squares sense) FIR filter.

    See Also
    --------
    firwin
    firwin2
    minimum_phase
    remez
    scipy.signal.firls
    """
    nyq = 0.5 * fs

    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError("numtaps must be odd and >= 1")
    M = (numtaps-1) // 2

    # normalize bands 0->1 and make it 2 columns
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = cupy.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError("bands must contain frequency pairs.")
    if (bands < 0).any() or (bands > 1).any():
        raise ValueError("bands must be between 0 and 1 relative to Nyquist")
    bands.shape = (-1, 2)

    # check remaining params
    desired = cupy.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError("desired must have one entry per frequency, got %s "
                         "gains for %s frequencies."
                         % (desired.size, bands.size))
    desired.shape = (-1, 2)
    # if (cupy.diff(bands) <= 0).any() or (cupy.diff(bands[:, 0]) < 0).any():
    #     raise ValueError("bands must be monotonically nondecreasing and have"
    #                     " width > 0.")
    # if (bands[:-1, 1] > bands[1:, 0]).any():
    #     raise ValueError("bands must not overlap.")
    # if (desired < 0).any():
    #     raise ValueError("desired must be non-negative.")
    if weight is None:
        weight = cupy.ones(len(desired))
    weight = cupy.asarray(weight).flatten()
    if len(weight) != len(desired):
        raise ValueError("weight must be the same size as the number of "
                         "band pairs ({}).".format(len(bands)))
    # if (weight < 0).any():
    #    raise ValueError("weight must be non-negative.")

    # Set up the linear matrix equation to be solved, Qa = b

    # We can express Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
    # where Q1(k,n)=q(k-n) and Q2(k,n)=q(k+n), i.e. a Toeplitz plus Hankel.

    # We omit the factor of 0.5 above, instead adding it during coefficient
    # calculation.

    # We also omit the 1/π from both Q and b equations, as they cancel
    # during solving.

    # We have that:
    #     q(n) = 1/π ∫W(ω)cos(nω)dω (over 0->π)
    # Using our nomalization ω=πf and with a constant weight W over each
    # interval f1->f2 we get:
    #     q(n) = W∫cos(πnf)df (0->1) = Wf sin(πnf)/πnf
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = cupy.arange(numtaps)[:, cupy.newaxis, cupy.newaxis]
    q = cupy.dot(cupy.diff(cupy.sinc(bands * n) *
                           bands, axis=2)[:, :, 0], weight)

    # Now we assemble our sum of Toeplitz and Hankel
    Q1 = toeplitz(q[:M+1])
    Q2 = hankel(q[:M+1], q[M:])
    Q = Q1 + Q2

    # Now for b(n) we have that:
    #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (over 0->π)
    # Using our normalization ω=πf and with a constant weight W over each
    # interval and a linear term for D(ω) we get (over each f1->f2 interval):
    #     b(n) = W ∫ (mf+c)cos(πnf)df
    #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = n[:M + 1]  # only need this many coefficients here
    # Choose m and c such that we are at the start and end weights
    m = (cupy.diff(desired, axis=1) / cupy.diff(bands, axis=1))
    c = desired[:, [0]] - bands[:, [0]] * m
    b = bands * (m*bands + c) * cupy.sinc(bands * n)
    # Use L'Hospital's rule here for cos(nπf)/(πnf)**2 @ n=0
    b[0] -= m * bands * bands / 2.
    b[1:] += m * cupy.cos(n[1:] * cupy.pi * bands) / (cupy.pi * n[1:]) ** 2
    b = cupy.diff(b, axis=2)[:, :, 0] @ weight

    # Now we can solve the equation : XXX CuPy failure modes (?)
    with cupyx.errstate(linalg="raise"):
        try:
            a = solve(Q, b)
        except LinAlgError:
            # in case Q is rank deficient
            a = lstsq(Q, b, rcond=None)[0]

    # XXX: scipy.signal does this:
    # try:  # try the fast way
    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter('always')
    #         a = solve(Q, b)
    #     for ww in w:
    #         if (ww.category == LinAlgWarning and
    #                 str(ww.message).startswith('Ill-conditioned matrix')):
    #             raise LinAlgError(str(ww.message))
    # except LinAlgError:  # in case Q is rank deficient
    #     a = lstsq(Q, b)[0]

    # make coefficients symmetric (linear phase)
    coeffs = cupy.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs
