"""Functions for FIR filter design."""

from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx

import cupy

__all__ = ["firls", "minimum_phase"]


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


def _dhtm(mag):
    """Compute the modified 1-D discrete Hilbert transform

    Parameters
    ----------
    mag : ndarray
        The magnitude spectrum. Should be 1-D with an even length, and
        preferably a fast length for FFT/IFFT.
    """
    # Adapted based on code by Niranjan Damera-Venkata,
    # Brian L. Evans and Shawn R. McCaslin (see refs for `minimum_phase`)
    sig = cupy.zeros(len(mag))
    # Leave Nyquist and DC at 0, knowing np.abs(fftfreq(N)[midpt]) == 0.5
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt + 1:] = -1
    # eventually if we want to support complex filters, we will need a
    # cupy.abs() on the mag inside the log, and should remove the .real
    recon = ifft(mag * cupy.exp(fft(sig * ifft(cupy.log(mag))))).real
    return recon


def minimum_phase(h, method='homomorphic', n_fft=None):
    """Convert a linear-phase FIR filter to minimum phase

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    method : {'hilbert', 'homomorphic'}
        The method to use:

            'homomorphic' (default)
                This method [4]_ [5]_ works best with filters with an
                odd number of taps, and the resulting minimum phase filter
                will have a magnitude response that approximates the square
                root of the original filter's magnitude response.

            'hilbert'
                This method [1]_ is designed to be used with equiripple
                filters (e.g., from `remez`) with unity or zero gain
                regions.

    n_fft : int
        The number of points to use for the FFT. Should be at least a
        few times larger than the signal length (see Notes).

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.

    See Also
    --------
    scipy.signal.minimum_phase

    Notes
    -----
    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
    of an FFT length to estimate the complex cepstrum of the filter.

    In the case of the Hilbert method, the deviation from the ideal
    spectrum ``epsilon`` is related to the number of stopband zeros
    ``n_stop`` and FFT length ``n_fft`` as::

        epsilon = 2. * n_stop / n_fft

    For example, with 100 stopband zeros and a FFT length of 2048,
    ``epsilon = 0.0976``. If we conservatively assume that the number of
    stopband zeros is one less than the filter length, we can take the FFT
    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::

        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))

    This gives reasonable results for both the Hilbert and homomorphic
    methods, and gives the value used when ``n_fft=None``.

    Alternative implementations exist for creating minimum-phase filters,
    including zero inversion [2]_ and spectral factorization [3]_ [4]_ [5]_.
    For more information, see:

        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters

    References
    ----------
    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
           complex minimum phase digital FIR filters," Acoustics, Speech,
           and Signal Processing, 1999. Proceedings., 1999 IEEE International
           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
           DOI:10.1109/ICASSP.1999.756179
    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
           filters by direct factorization," Signal Processing,
           vol. 10, no. 4, pp. 369-383, Jun. 1986.
    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
           Handbook for Digital Signal Processing, chapter 4,
           New York: Wiley-Interscience, 1993.
    .. [4] J. S. Lim, Advanced Topics in Signal Processing.
           Englewood Cliffs, N.J.: Prentice Hall, 1988.
    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
           "Discrete-Time Signal Processing," 2nd edition.
           Upper Saddle River, N.J.: Prentice Hall, 1999.

    """  # noqa
    if cupy.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    n_half = len(h) // 2
    if not cupy.allclose(h[-n_half:][::-1], h[:n_half]):
        import warnings
        warnings.warn('h does not appear to by symmetric, conversion may '
                      'fail', RuntimeWarning)
    if not isinstance(method, str) or method not in \
            ('homomorphic', 'hilbert',):
        raise ValueError('method must be "homomorphic" or "hilbert", got %r'
                         % (method,))
    if n_fft is None:
        n_fft = 2 ** int(cupy.ceil(cupy.log2(2 * (len(h) - 1) / 0.01)))
    n_fft = int(n_fft)
    if n_fft < len(h):
        raise ValueError('n_fft must be at least len(h)==%s' % len(h))
    if method == 'hilbert':
        w = cupy.arange(n_fft) * (2 * cupy.pi / n_fft * n_half)
        H = cupy.real(fft(h, n_fft) * cupy.exp(1j * w))
        dp = max(H) - 1
        ds = 0 - min(H)
        S = 4. / (cupy.sqrt(1 + dp + ds) + cupy.sqrt(1 - dp + ds)) ** 2
        H += ds
        H *= S
        H = cupy.sqrt(H, out=H)
        H += 1e-10  # ensure that the log does not explode
        h_minimum = _dhtm(H)
    else:  # method == 'homomorphic'
        # zero-pad; calculate the DFT
        h_temp = cupy.abs(fft(h, n_fft))
        # take 0.25*log(|H|**2) = 0.5*log(|H|)
        h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
        cupy.log(h_temp, out=h_temp)
        h_temp *= 0.5
        # IDFT
        h_temp = ifft(h_temp).real
        # multiply pointwise by the homomorphic filter
        # lmin[n] = 2u[n] - d[n]
        win = cupy.zeros(n_fft)
        win[0] = 1
        stop = (len(h) + 1) // 2
        win[1:stop] = 2
        if len(h) % 2:
            win[stop] = 1
        h_temp *= win
        h_temp = ifft(cupy.exp(fft(h_temp)))
        h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]
