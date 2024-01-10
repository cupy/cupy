import cupy
import cupyx.scipy.linalg
import cupyx.scipy.signal._signaltools
from cupyx.scipy.signal._arraytools import (
    axis_slice,
    axis_reverse,
    const_ext,
    even_ext,
    odd_ext,
)


def firfilter(b, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an FIR filter.

    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
    for large filtering operations (and inherently depends on fftconvolve)

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """

    return cupyx.scipy.signal.lfilter(b, cupy.ones((1,)), x, axis, zi)


def firfilter_zi(b):
    """
    Construct initial conditions for lfilter for step response steady-state.
    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.
    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.
    Parameters
    ----------
    b : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.
    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.
    See Also
    --------
    lfilter, lfiltic, filtfilt
    Notes
    -----
    A linear filter with order m has a state space representation (A, B, C, D),
    for which the output y of the filter can be expressed as::
        z(n+1) = A*z(n) + B*x(n)
        y(n)   = C*z(n) + D*x(n)
    where z(n) is a vector of length m, A has shape (m, m), B has shape
    (m, 1), C has shape (1, m) and D has shape (1, 1) (assuming x(n) is
    a scalar).  lfilter_zi solves::
        zi = A*zi + B
    In other words, it finds the initial condition for which the response
    to an input of all ones is a constant.
    Given the filter coefficients `a` and `b`, the state space matrices
    for the transposed direct form II implementation of the linear filter,
    which is the implementation used by scipy.signal.lfilter, are::
        A = scipy.linalg.companion(a).T
        B = b[1:] - a[1:]*b[0]
    assuming `a[0]` is 1.0; if `a[0]` is not 1, `a` and `b` are first
    divided by a[0].
    """
    a = cupy.ones(1, dtype=cupy.float32)
    n = len(b)

    # Pad a with zeros so they are the same length.
    a = cupy.r_[a, cupy.zeros(n - len(a), dtype=a.dtype)]
    IminusA = (cupy.eye(n - 1, dtype=cupy.result_type(a, b))
               - cupyx.scipy.linalg.companion(a).T)
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = cupy.linalg.solve(IminusA, B)

    return zi


def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ["even", "odd", "constant", None]:
        raise ValueError(
            (
                "Unknown value '%s' given to padtype.  padtype "
                "must be 'even', 'odd', 'constant', or None."
            )
            % padtype
        )

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError(
            "The length of the input vector x must be greater "
            "than padlen, which is %d." % edge
        )

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == "even":
            ext = even_ext(x, edge, axis=axis)
        elif padtype == "odd":
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext


def firfilter2(
    b, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None
):
    """
    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.
    The function provides options for handling the edges of the signal.
    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.
    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.
    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.
    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.
    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.
    The option to use Gustaffson's method was added in scipy version 0.16.0.
    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
    b = cupy.atleast_1d(b)
    x = cupy.asarray(x)
    if method not in ["pad", "gust"]:
        raise ValueError("method must be 'pad' or 'gust'.")

    if method == "gust":
        raise NotImplementedError("gust method not supported yet")

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=len(b))

    # Get the steady state of the filter's step response.
    zi = firfilter_zi(b)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = cupy.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = firfilter(b, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = firfilter(b, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return cupy.copy(y)
