
import cupy
from cupy.linalg import solve
from cupyx.scipy.interpolate._interpolate import PPoly
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
import cupyx.scipy.special as spec


def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return cupy.isscalar(x) or hasattr(x, 'shape') and x.shape == ()


def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.
    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).
    """

    x, y = map(cupy.asarray, (x, y))
    if cupy.issubdtype(x.dtype, cupy.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    if cupy.issubdtype(y.dtype, cupy.complexfloating):
        dtype = complex
    else:
        dtype = float

    if dydx is not None:
        dydx = cupy.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if cupy.issubdtype(dydx.dtype, cupy.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError("The length of `y` along `axis`={0} doesn't "
                         "match the length of `x`".format(axis))

    if not cupy.all(cupy.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not cupy.all(cupy.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not cupy.all(cupy.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = cupy.diff(x)
    if cupy.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = cupy.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = cupy.moveaxis(dydx, axis, 0)

    return x, dx, y, axis, dydx


class CubicHermiteSpline(PPoly):
    """Piecewise-cubic interpolator matching values and first derivatives.

    The result is represented as a `PPoly` instance. [1]_

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    dydx : array_like
        Array containing derivatives of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``cupy.take(y, i, axis=axis)``.
        Default is 0.
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), it is set to True.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-D, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.

    See Also
    --------
    Akima1DInterpolator : Akima 1D interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    If you want to create a higher-order spline matching higher-order
    derivatives, use `BPoly.from_derivatives`.

    References
    ----------
    .. [1] `Cubic Hermite spline
            <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_
            on Wikipedia.
    """

    def __init__(self, x, y, dydx, axis=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = True

        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = cupy.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

        c = cupy.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - dydx[:-1]) / dxr - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]

        super().__init__(c, x, extrapolate=extrapolate)
        self.axis = axis


class PchipInterpolator(CubicHermiteSpline):
    r"""PCHIP 1-D monotonic cubic interpolation.

    ``x`` and ``y`` are arrays of values used to approximate some function f,
    with ``y = f(x)``. The interpolant uses monotonic cubic splines
    to find the value of new points. (PCHIP stands for Piecewise Cubic
    Hermite Interpolating Polynomial).

    Parameters
    ----------
    x : ndarray
        A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
    y : ndarray
        A 1-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. If N-D array, use ``axis``
        parameter to select correct axis.
    axis : int, optional
        Axis in the y array corresponding to the x-coordinate values.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    Akima1DInterpolator : Akima 1D interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.

    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at :math:`x_k`.

    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
    by using PCHIP algorithm [1]_.

    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
    are the slopes at internal points :math:`x_k`.
    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
    weighted harmonic mean

    .. math::

        \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

    The end slopes are set using a one-sided scheme [2]_.


    References
    ----------
    .. [1] F. N. Fritsch and J. Butland,
           A method for constructing local
           monotone piecewise cubic interpolants,
           SIAM J. Sci. Comput., 5(2), 300-304 (1984).
           `10.1137/0905021 <https://doi.org/10.1137/0905021>`_.
    .. [2] see, e.g., C. Moler, Numerical Computing with Matlab, 2004.
           `10.1137/1.9780898717952 <https://doi.org/10.1137/1.9780898717952>`_
    """

    def __init__(self, x, y, axis=0, extrapolate=None):
        x, _, y, axis, _ = prepare_input(x, y, axis)
        xp = x.reshape((x.shape[0],) + (1,)*(y.ndim-1))
        dk = self._find_derivatives(xp, y)
        super().__init__(x, y, dk, axis=0, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _edge_case(h0, h1, m0, m1):
        # one-sided three-point estimate for the derivative
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

        # try to preserve shape
        mask = cupy.sign(d) != cupy.sign(m0)
        mask2 = (cupy.sign(m0) != cupy.sign(m1)) & (
            cupy.abs(d) > 3.*cupy.abs(m0))
        mmm = (~mask) & mask2

        d[mask] = 0.
        d[mmm] = 3.*m0[mmm]

        return d

    @staticmethod
    def _find_derivatives(x, y):
        # Determine the derivatives at the points y_k, d_k, by using
        #  PCHIP algorithm is:
        # We choose the derivatives at the point x_k by
        # Let m_k be the slope of the kth segment (between k and k+1)
        # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
        # else use weighted harmonic mean:
        #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
        #   where h_k is the spacing between x_k and x_{k+1}
        y_shape = y.shape
        if y.ndim == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            y = y[:, None]

        hk = x[1:] - x[:-1]
        mk = (y[1:] - y[:-1]) / hk

        if y.shape[0] == 2:
            # edge case: only have two points, use linear interpolation
            dk = cupy.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return dk.reshape(y_shape)

        smk = cupy.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)

        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]

        # values where division by zero occurs will be excluded
        # by 'condition' afterwards
        whmean = (w1 / mk[:-1] + w2 / mk[1:]) / (w1 + w2)

        dk = cupy.zeros_like(y)
        dk[1:-1] = cupy.where(condition, 0.0, 1.0 / whmean)

        # special case endpoints, as suggested in
        # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
        dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return dk.reshape(y_shape)


def pchip_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for pchip interpolation.

    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``. The interpolant uses monotonic cubic splines
    to find the value of new points x and the derivatives there.
    See `scipy.interpolate.PchipInterpolator` for details.

    Parameters
    ----------
    xi : array_like
        A sorted list of x-coordinates, of length N.
    yi : array_like
        A 1-D array of real values. `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.
    x : scalar or array_like
        Of length M.
    der : int or list, optional
        Derivatives to extract. The 0th derivative can be included to
        return the function value.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.

    Returns
    -------
    y : scalar or array_like
        The result, of length R or length M or M by R.
    """
    P = PchipInterpolator(xi, yi, axis=axis)

    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(der)(x)
    else:
        return [P.derivative(nu)(x) for nu in der]


class Akima1DInterpolator(CubicHermiteSpline):
    """
    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural [1]_.

    Parameters
    ----------
    x : ndarray, shape (m, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (m, ...)
        N-D array of real values. The length of ``y`` along the first axis
        must be equal to the length of ``x``.
    axis : int, optional
        Specifies the axis of ``y`` along which to interpolate. Interpolation
        defaults to the first axis of ``y``.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    References
    ----------
    .. [1] A new method of interpolation and smooth curve fitting based
        on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
        589-602.
    """

    def __init__(self, x, y, axis=0):
        # Original implementation in MATLAB by N. Shamsundar (BSD licensed)
        # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation  # noqa: E501
        x, dx, y, axis, _ = prepare_input(x, y, axis)

        # determine slopes between breakpoints
        m = cupy.empty((x.size + 3, ) + y.shape[1:])
        dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
        m[2:-2] = cupy.diff(y, axis=0) / dx

        # add two additional points on the left ...
        m[1] = 2. * m[2] - m[3]
        m[0] = 2. * m[1] - m[2]
        # ... and on the right
        m[-2] = 2. * m[-3] - m[-4]
        m[-1] = 2. * m[-2] - m[-3]

        # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
        # defined. This is the fill value:
        t = .5 * (m[3:] + m[:-3])
        # get the denominator of the slope t
        dm = cupy.abs(cupy.diff(m, axis=0))
        f1 = dm[2:]
        f2 = dm[:-2]
        f12 = f1 + f2
        # These are the mask of where the slope at breakpoint is defined:
        max_value = -cupy.inf if y.size == 0 else cupy.max(f12)
        ind = cupy.nonzero(f12 > 1e-9 * max_value)
        x_ind, y_ind = ind[0], ind[1:]
        # Set the slope at breakpoint
        t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
                  f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]

        super().__init__(x, y, t, axis=0, extrapolate=False)
        self.axis = axis

    def extend(self, c, x, right=True):
        raise NotImplementedError("Extending a 1-D Akima interpolator is not "
                                  "yet implemented")

    # These are inherited from PPoly, but they do not produce an Akima
    # interpolator. Hence stub them out.
    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        raise NotImplementedError("This method does not make sense for "
                                  "an Akima interpolator.")


def _validate_bc(bc_type, y, expected_deriv_shape, axis):
    """Validate and prepare boundary conditions.

    Returns
    -------
    validated_bc : 2-tuple
        Boundary conditions for a curve start and end.
    y : ndarray
        y casted to complex dtype if one of the boundary conditions has
        complex dtype.
    """
    if isinstance(bc_type, str):
        if bc_type == 'periodic':
            if not cupy.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                raise ValueError(
                    f"The first and last `y` point along axis {axis} must "
                    "be identical (within machine precision) when "
                    "bc_type='periodic'.")

        bc_type = (bc_type, bc_type)

    else:
        if len(bc_type) != 2:
            raise ValueError("`bc_type` must contain 2 elements to "
                             "specify start and end conditions.")

        if 'periodic' in bc_type:
            raise ValueError("'periodic' `bc_type` is defined for both "
                             "curve ends and cannot be used with other "
                             "boundary conditions.")

    validated_bc = []
    for bc in bc_type:
        if isinstance(bc, str):
            if bc == 'clamped':
                validated_bc.append((1, cupy.zeros(expected_deriv_shape)))
            elif bc == 'natural':
                validated_bc.append((2, cupy.zeros(expected_deriv_shape)))
            elif bc in ['not-a-knot', 'periodic']:
                validated_bc.append(bc)
            else:
                raise ValueError(f"bc_type={bc} is not allowed.")
        else:
            try:
                deriv_order, deriv_value = bc
            except Exception as e:
                raise ValueError(
                    "A specified derivative value must be "
                    "given in the form (order, value)."
                ) from e

            if deriv_order not in [1, 2]:
                raise ValueError("The specified derivative order must "
                                 "be 1 or 2.")

            deriv_value = cupy.asarray(deriv_value)
            if deriv_value.shape != expected_deriv_shape:
                raise ValueError(
                    "`deriv_value` shape {} is not the expected one {}."
                    .format(deriv_value.shape, expected_deriv_shape))

            if cupy.issubdtype(deriv_value.dtype, cupy.complexfloating):
                y = y.astype(complex, copy=False)

            validated_bc.append((deriv_order, deriv_value))

    return validated_bc, y

# XXX: upstream to scipy? (careful with splPrep)


def _from_spline(spl):
    """PPoly.from_spline replacement which handles y.ndim > 1."""
    t, c, k = spl.tck
    axis = spl.axis
    cvals = cupy.empty((k+1, len(t)-1) + c.shape[1:], dtype=c.dtype)

    # convert: here axis=0 because spl(x) rolls the interpolation axis back
    for m in range(k, -1, -1):
        ym = spl(t[:-1], nu=m)
        ym = cupy.moveaxis(ym, axis, 0)
        cvals[k - m, ...] = ym / spec.gamma(m+1)

    # redo the axis reshuffle in _PPolyBase.__init__:
    # https://github.com/scipy/scipy/blob/v1.12.0/scipy/interpolate/_interpolate.py#L826
    cvals_ = cupy.moveaxis(cvals, 0, axis+1)
    cvals_ = cupy.moveaxis(cvals_, 0, axis+1)

    pp = PPoly(cvals_, t, axis=axis)
    return pp


class CubicSpline(CubicHermiteSpline):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like, shape (n,)
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:

        * ``order``: the derivative order, 1 or 2.
        * ``deriv_value``: array_like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1-D, then `deriv_value` must be a scalar. If `y` is 3-D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2-D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.

    See Also
    --------
    scipy.interpolate.CubicSpline

    Notes
    -----
    Parameters `bc_type` and ``extrapolate`` work independently, i.e. the
    former controls only construction of a spline, and the latter only
    evaluation.

    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.

    """

    def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
        x = cupy.asarray(x)
        y = cupy.asarray(y)

        if y.size == 0:
            # bail out early for zero-sized arrays
            s = cupy.zeros_like(y)
            super().__init__(x, y, s, axis=axis, extrapolate=extrapolate)
            self.axis = axis
        elif len(x) <= 3:
            # special cases: make_interp_spline requires >k data points
            # vendor what scipy.interpolate.CubicSpline does
            x, dx, y, axis, _ = prepare_input(x, y, axis)
            n = len(x)

            bc, y = _validate_bc(bc_type, y, y.shape[1:], axis)

            if extrapolate is None:
                if bc[0] == 'periodic':
                    extrapolate = 'periodic'
                else:
                    extrapolate = True

            dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
            slope = cupy.diff(y, axis=0) / dxr

            # If bc is 'not-a-knot' this change is just a convention.
            # If bc is 'periodic' then we already checked that y[0] == y[-1],
            # and the spline is just a constant, we handle this case in the
            # same way by setting the first derivatives to slope, which is 0.
            if n == 2:
                if bc[0] in ['not-a-knot', 'periodic']:
                    bc[0] = (1, slope[0])
                if bc[1] in ['not-a-knot', 'periodic']:
                    bc[1] = (1, slope[0])
                s = cupy.r_[slope, slope]

            # This is a special case, when both conditions are 'not-a-knot'
            # and n == 3. In this case 'not-a-knot' can't be handled regularly
            # as the both conditions are identical. We handle this case by
            # constructing a parabola passing through given points.
            if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
                A = cupy.zeros((3, 3))  # This is a standard matrix.
                b = cupy.empty((3,) + y.shape[1:], dtype=y.dtype)

                A[0, 0] = 1
                A[0, 1] = 1
                A[1, 0] = dx[1]
                A[1, 1] = 2 * (dx[0] + dx[1])
                A[1, 2] = dx[0]
                A[2, 1] = 1
                A[2, 2] = 1

                b[0] = 2 * slope[0]
                b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
                b[2] = 2 * slope[1]

                s = solve(A, b)
            elif n == 3 and bc[0] == 'periodic':
                # In case when number of points is 3 we compute the derivatives
                # manually
                t = (slope / dxr).sum(0) / (1. / dxr).sum(0)
                s = cupy.broadcast_to(t, (n,) + y.shape[1:])

            # finally, construct the object
            super().__init__(x, y, s, axis=0, extrapolate=extrapolate)
            self.axis = axis
        else:
            # general case: delegate to make_interp_spline and convert to PPoly
            # first, repackage the boundary conditions.
            # Also account for that complex b.c. upcast y in CubicSpline
            need_complex = False
            if not isinstance(bc_type, str):
                # must be a 2-tuple
                bc_0, bc_1 = bc_type
                if not isinstance(bc_0, str):
                    need_complex = cupy.iscomplexobj(bc_0[1])
                    bc_0 = [bc_0]
                if not isinstance(bc_1, str):
                    need_complex = cupy.iscomplexobj(bc_1[1])
                    bc_1 = [bc_1]
                bc_type = (bc_0, bc_1)

            if need_complex:
                y = y.astype(complex)

            # actually do the work
            spl = make_interp_spline(x, y, k=3, axis=axis, bc_type=bc_type)
            pp = _from_spline(spl)
            self.x = pp.x
            self.c = pp.c
            self.axis = pp.axis

            if extrapolate is None:
                extrapolate = True
            self.extrapolate = extrapolate
