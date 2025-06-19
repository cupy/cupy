import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial


def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return cupy.isscalar(x) or hasattr(x, 'shape') and x.shape == ()


class _Interpolator1D:
    """Common features in univariate interpolation.

    Deal with input data type and interpolation axis rolling. The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis : Axis along which the interpolation goes in the
        original array
    _y_extra_shape : Additional shape of the input arrays, excluding
        the interpolation axis
    dtype : Dtype of the y-data arrays. It can be set via _set_dtype,
        which forces it to be float or complex

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """Evaluate the interpolant

        Parameters
        ----------
        x : cupy.ndarray
            The points to evaluate the interpolant

        Returns
        -------
        y : cupy.ndarray
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x

        Notes
        -----
        Input values `x` must be convertible to `float` values like `int`
        or `float`.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """
        Reshape input array to 1-D
        """
        x = cupy.asarray(x)
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """
        Reshape interpolated y back to an N-D array similar to initial y
        """
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx + self._y_axis, nx + ny)))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        """
        Reshape the updated yi to a 1-D array
        """
        yi = cupy.moveaxis(yi, self._y_axis, 0)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "%r + (N,) + %r" % (self._y_extra_shape[-self._y_axis:],
                                           self._y_extra_shape[:-self._y_axis])
            raise ValueError("Data must be of shape %s" % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")

        self._y_axis = (axis % yi.ndim)
        self._y_extra_shape = yi.shape[:self._y_axis]+yi.shape[self._y_axis+1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if cupy.issubdtype(dtype, cupy.complexfloating) \
                or cupy.issubdtype(self.dtype, cupy.complexfloating):
            self.dtype = cupy.complex128
        else:
            if not union or self.dtype != cupy.complex128:
                self.dtype = cupy.float64


class _Interpolator1DWithDerivatives(_Interpolator1D):

    def derivatives(self, x, der=None):
        """Evaluate many derivatives of the polynomial at the point x.

        The function produce an array of all derivative values at
        the point x.

        Parameters
        ----------
        x : cupy.ndarray
            Point or points at which to evaluate the derivatives
        der : int or None, optional
            How many derivatives to extract; None for all potentially
            nonzero derivatives (that is a number equal to the number
            of points). This number includes the function value as 0th
            derivative

        Returns
        -------
        d : cupy.ndarray
            Array with derivatives; d[j] contains the jth derivative.
            Shape of d[j] is determined by replacing the interpolation
            axis in the original array with the shape of x

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der)

        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = ([0] + list(range(nx+1, nx + self._y_axis+1))
                 + list(range(1, nx+1)) +
                 list(range(nx+1+self._y_axis, nx+ny+1)))
            y = y.transpose(s)
        return y

    def derivative(self, x, der=1):
        """Evaluate one derivative of the polynomial at the point x

        Parameters
        ----------
        x : cupy.ndarray
            Point or points at which to evaluate the derivatives
        der : integer, optional
            Which derivative to extract. This number includes the
            function value as 0th derivative

        Returns
        -------
        d : cupy.ndarray
            Derivative interpolated at the x-points. Shape of d is
            determined by replacing the interpolation axis in the
            original array with the shape of x

        Notes
        -----
        This is computed by evaluating all derivatives up to the desired
        one (using self.derivatives()) and then discarding the rest.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der+1)
        return self._finish_y(y[der], x_shape)


class BarycentricInterpolator(_Interpolator1D):
    """The interpolating polynomial for a set of points.

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial, efficient changing of the y
    values to be interpolated, and updating by adding more x values.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.
    The value `yi` need to be provided before the function is
    evaluated, but none of the preprocessing depends on them,
    so rapid updates are possible.

    Parameters
    ----------
    xi : cupy.ndarray
        1-D array of x-coordinates of the points the polynomial should
        pass through
    yi : cupy.ndarray, optional
        The y-coordinates of the points the polynomial should pass through.
        If None, the y values will be supplied later via the `set_y` method
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values

    See Also
    --------
    scipy.interpolate.BarycentricInterpolator

    """

    def __init__(self, xi, yi=None, axis=0):
        _Interpolator1D.__init__(self, xi, yi, axis)

        self.xi = xi.astype(cupy.float64)
        self.set_yi(yi)
        self.n = len(self.xi)

        self._inv_capacity = 4.0 / (cupy.max(self.xi) - cupy.min(self.xi))
        permute = cupy.random.permutation(self.n)
        inv_permute = cupy.zeros(self.n, dtype=cupy.int32)
        inv_permute[permute] = cupy.arange(self.n)

        self.wi = cupy.zeros(self.n)
        for i in range(self.n):
            dist = self._inv_capacity * (self.xi[i] - self.xi[permute])
            dist[inv_permute[i]] = 1.0
            self.wi[i] = 1.0 / cupy.prod(dist)

    def set_yi(self, yi, axis=None):
        """Update the y values to be interpolated.

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the xi. The yi can be changed
        at any time.

        Parameters
        ----------
        yi : cupy.ndarray
            The y-coordinates of the points the polynomial should pass
            through. If None, the y values will be supplied later.
        axis : int, optional
            Axis in the yi array corresponding to the x-coordinate values

        """

        if yi is None:
            self.yi = None
            return
        self._set_yi(yi, xi=self.xi, axis=axis)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

    def add_xi(self, xi, yi=None):
        """Add more x values to the set to be interpolated.

        The barycentric interpolation algorithm allows easy updating
        by adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : cupy.ndarray
            The x-coordinates of the points that the polynomial should
            pass through
        yi : cupy.ndarray, optional
            The y-coordinates of the points the polynomial should pass
            through. Should have shape ``(xi.size, R)``; if R > 1 then
            the polynomial is vector-valued
            If `yi` is not given, the y values will be supplied later.
            `yi` should be given if and only if the interpolator has y
            values specified

        """

        if yi is not None:
            if self.yi is None:
                raise ValueError("No previous yi value to update!")
            yi = self._reshape_yi(yi, check=True)
            self.yi = cupy.vstack((self.yi, yi))
        else:
            if self.yi is not None:
                raise ValueError("No update to yi provided!")
        old_n = self.n
        self.xi = cupy.concatenate((self.xi, xi))
        self.n = len(self.xi)
        self.wi **= -1
        old_wi = self.wi
        self.wi = cupy.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n, self.n):
            self.wi[:j] *= self._inv_capacity * (self.xi[j] - self.xi[:j])
            self.wi[j] = cupy.prod(
                self._inv_capacity * (self.xi[:j] - self.xi[j])
            )
        self.wi **= -1

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x.

        Parameters
        ----------
        x : cupy.ndarray
            Points to evaluate the interpolant at

        Returns
        -------
        y : cupy.ndarray
            Interpolated values. Shape is determined by replacing the
            interpolation axis in the original array with the shape of x

        Notes
        -----
        Currently the code computes an outer product between x and the
        weights, that is, it constructs an intermediate array of size
        N by len(x), where N is the degree of the polynomial.

        """

        return super().__call__(x)

    def _evaluate(self, x):
        if x.size == 0:
            p = cupy.zeros((0, self.r), dtype=self.dtype)
        else:
            c = x[..., cupy.newaxis] - self.xi
            z = c == 0
            c[z] = 1
            c = self.wi / c
            p = cupy.dot(c, self.yi) / cupy.sum(c, axis=-1)[..., cupy.newaxis]
            r = cupy.nonzero(z)
            if len(r) == 1:  # evaluation at a scalar
                if len(r[0]) > 0:  # equals one of the points
                    p = self.yi[r[0][0]]
            else:
                p[r[:-1]] = self.yi[r[-1]]
        return p


def barycentric_interpolate(xi, yi, x, axis=0):
    """Convenience function for polynomial interpolation.

    Constructs a polynomial that passes through a given
    set of points, then evaluates the polynomial. For
    reasons of numerical stability, this function does
    not compute the coefficients of the polynomial.

    Parameters
    ----------
    xi : cupy.ndarray
        1-D array of coordinates of the points the polynomial
        should pass through
    yi : cupy.ndarray
        y-coordinates of the points the polynomial should pass
        through
    x : scalar or cupy.ndarray
        Points to evaluate the interpolator at
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate
        values

    Returns
    -------
    y : scalar or cupy.ndarray
        Interpolated values. Shape is determined by replacing
        the interpolation axis in the original array with the
        shape x

    See Also
    --------
    scipy.interpolate.barycentric_interpolate

    """

    return BarycentricInterpolator(xi, yi, axis=axis)(x)


class KroghInterpolator(_Interpolator1DWithDerivatives):
    """Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs (xi,yi). One may
    additionally specify a number of derivatives at each point xi;
    this is done by repeating the value xi and specifying the
    derivatives as successive yi values
    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : cupy.ndarray, length N
        x-coordinate, must be sorted in increasing order
    yi : cupy.ndarray
        y-coordinate, when a xi occurs two or more times in a row,
        the corresponding yi's represent derivative values
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    """

    def __init__(self, xi, yi, axis=0):
        _Interpolator1DWithDerivatives.__init__(self, xi, yi, axis)

        self.xi = xi.astype(cupy.float64)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

        c = cupy.zeros((self.n+1, self.r), dtype=self.dtype)
        c[0] = self.yi[0]
        Vk = cupy.zeros((self.n, self.r), dtype=self.dtype)
        for k in range(1, self.n):
            s = 0
            while s <= k and xi[k-s] == xi[k]:
                s += 1
            s -= 1
            Vk[0] = self.yi[k]/float_factorial(s)
            for i in range(k-s):
                if xi[i] == xi[k]:
                    raise ValueError("Elements if `xi` can't be equal.")
                if s == 0:
                    Vk[i+1] = (c[i]-Vk[i])/(xi[i]-xi[k])
                else:
                    Vk[i+1] = (Vk[i+1]-Vk[i])/(xi[i]-xi[k])
            c[k] = Vk[k-s]
        self.c = c

    def _evaluate(self, x):
        pi = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]
        for k in range(1, self.n):
            w = x - self.xi[k-1]
            pi = w*pi
            p += pi[:, cupy.newaxis] * self.c[k]
        return p

    def _evaluate_derivatives(self, x, der=None):
        n = self.n
        r = self.r

        if der is None:
            der = self.n
        pi = cupy.zeros((n, len(x)))
        w = cupy.zeros((n, len(x)))
        pi[0] = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]

        for k in range(1, n):
            w[k-1] = x - self.xi[k-1]
            pi[k] = w[k-1] * pi[k-1]
            p += pi[k, :, cupy.newaxis] * self.c[k]

        cn = cupy.zeros((max(der, n+1), len(x), r), dtype=self.dtype)
        cn[:n+1, :, :] += self.c[:n+1, cupy.newaxis, :]
        cn[0] = p
        for k in range(1, n):
            for i in range(1, n-k+1):
                pi[i] = w[k+i-1]*pi[i-1] + pi[i]
                cn[k] = cn[k] + pi[i, :, cupy.newaxis]*cn[k+i]
            cn[k] *= float_factorial(k)

        cn[n, :, :] = 0
        return cn[:der]


def krogh_interpolate(xi, yi, x, der=0, axis=0):
    """Convenience function for polynomial interpolation

    Parameters
    ----------
    xi : cupy.ndarray
        x-coordinate
    yi : cupy.ndarray
        y-coordinates, of shape ``(xi.size, R)``. Interpreted as
        vectors of length R, or scalars if R=1
    x : cupy.ndarray
        Point or points at which to evaluate the derivatives
    der : int or list, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values

    Returns
    -------
    d : cupy.ndarray
        If the interpolator's values are R-D then the
        returned array will be the number of derivatives by N by R.
        If `x` is a scalar, the middle dimension will be dropped; if
        the `yi` are scalars then the last dimension will be dropped

    See Also
    --------
    scipy.interpolate.krogh_interpolate

    """
    P = KroghInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=cupy.amax(der)+1)[der]


def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = cupy.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError(f'{name} argument must be able to broadcast up '
                     f'to shape {shape_to} but had shape {shape_from}')


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    return (isinstance(fill_value, str) and
            fill_value == 'extrapolate')


class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (npoints, ) array_like
        A 1-D array of real values.
    y : (..., npoints, ...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`. Use the ``axis`` parameter
        to select correct axis. Unlike other interpolators, the default
        interpolation axis is the last axis of `y`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values.
        Unlike other interpolators, defaults to ``axis=-1``.
    copy : bool, optional
        If ``True``, the class makes internal copies of x and y. If ``False``,
        references to ``x`` and ``y`` are used if possible. The default is to
        copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``. Using a two-element tuple
          or ndarray requires ``bounds_error=False``.
        - If "extrapolate", then points outside the data range will be
          extrapolated.

    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    See Also
    --------
    scipy.interpolate.interp1d

    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.

    If the values in `x` are not unique, the resulting behavior is
    undefined and specific to the choice of `kind`, i.e., changing
    `kind` will change the behavior for duplicates.
    """  # NOQA

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=cupy.nan,
                 assume_sorted=False):
        """ Initialize a 1-D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)

        self.bounds_error = bounds_error  # used by fill_value setter

        # `copy` keyword semantics changed in NumPy 2.0, once that is
        # the minimum version this can use `copy=None`.
        # XXX: https://github.com/scipy/scipy/blob/main/scipy/_lib/_util.py#L59
        self.copy = copy
        # if not copy:
        #    self.copy = copy_if_needed

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1,
                     'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest', 'nearest-up', 'previous',
                          'next'):
            raise NotImplementedError("%s is unsupported: Use fitpack "
                                      "routines for other types." % kind)
        x = cupy.array(x, copy=self.copy)
        y = cupy.array(y, copy=self.copy)

        if not assume_sorted:
            ind = cupy.argsort(x, kind="stable")  # scipy uses kind="mergesort"
            x = x[ind]
            y = cupy.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, cupy.inexact):
            y = y.astype(cupy.float64)

        # Backward compatibility
        self.axis = axis % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x  # clean up namespace to prevent misuse; use attributes
        self._kind = kind

        # Adjust to interpolation kind; store reference to *unbound*
        # interpolation methods, in order to avoid circular references to self
        # stored in the bound instance methods, and therefore delayed garbage
        # collection.  See: https://docs.python.org/reference/datamodel.html
        if kind in ('linear', 'nearest', 'nearest-up', 'previous', 'next'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            minval = 1
            if kind == 'nearest':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'left'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'nearest-up':
                # Do division before addition to prevent possible integer
                # overflow
                self._side = 'right'
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            elif kind == 'previous':
                # Side for cupy.searchsorted and index for clipping
                self._side = 'left'
                self._ind = 0
                # Move x by one floating point value to the left
                self._x_shift = cupy.nextafter(self.x, -cupy.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    # assume y is sorted by x ascending order here.
                    fill_value = (cupy.nan, cupy.take(self.y, -1, axis))
            elif kind == 'next':
                self._side = 'right'
                self._ind = 1
                # Move x by one floating point value to the right
                self._x_shift = cupy.nextafter(self.x, cupy.inf)
                self._call = self.__class__._call_previousnext
                if _do_extrapolate(fill_value):
                    self._check_and_update_bounds_error_for_extrapolation()
                    # assume y is sorted by x ascending order here.
                    fill_value = (cupy.take(self.y, 0, axis), cupy.nan)
            else:
                # Check if we can delegate to numpy.interp (2x-10x faster).
                np_dtypes = (cupy.dtype(cupy.float64), cupy.dtype(int))
                cond = self.x.dtype in np_dtypes and self.y.dtype in np_dtypes
                cond = cond and self.y.ndim == 1
                cond = cond and not _do_extrapolate(fill_value)

                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1

            rewrite_nan = False
            xx, yy = self.x, self._y
            if order > 1:
                # Quadratic or cubic spline. If input contains even a single
                # nan, then the output is all nans. We cannot just feed data
                # with nans to make_interp_spline because it calls LAPACK.
                # So, we make up a bogus x and y with no nans and use it
                # to get the correct shape of the output, which we then fill
                # with nans.
                # For slinear or zero order spline, we just pass nans through.
                mask = cupy.isnan(self.x)
                if mask.any():
                    sx = self.x[~mask]
                    if sx.size == 0:
                        raise ValueError("`x` array is all-nan")
                    xx = cupy.linspace(cupy.nanmin(self.x),
                                       cupy.nanmax(self.x),
                                       len(self.x))
                    rewrite_nan = True
                if cupy.isnan(self._y).any():
                    yy = cupy.ones_like(self._y)
                    rewrite_nan = True

            from cupyx.scipy.interpolate import make_interp_spline
            self._spline = make_interp_spline(xx, yy, k=order,
                                              check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline

        if len(self.x) < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)

        self.fill_value = fill_value  # calls the setter, can modify bounds_err

    @property
    def fill_value(self):
        """The fill value."""
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        # extrapolation only works for nearest neighbor and linear methods
        if _do_extrapolate(fill_value):
            self._check_and_update_bounds_error_for_extrapolation()
            self._extrapolate = True
        else:
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # it's either a pair (_below_range, _above_range) or a single value
            # for both above and below range
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [cupy.asarray(fill_value[0]),
                               cupy.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = cupy.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def _check_and_update_bounds_error_for_extrapolation(self):
        if self.bounds_error:
            raise ValueError("Cannot extrapolate and raise "
                             "at the same time.")
        self.bounds_error = False

    def _call_linear_np(self, x_new):
        # Note that out-of-bounds values are taken care of in self._evaluate
        return cupy.interp(x_new, self.x, self.y)

    def _call_linear(self, x_new):
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = cupy.searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbor interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbor
        x_new_indices = cupy.searchsorted(self.x_bds, x_new, side=self._side)

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(cupy.intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices]

        return y_new

    def _call_previousnext(self, x_new):
        """Use previous/next neighbor of x_new, y_new = f(x_new)."""

        # 1. Get index of left/right value
        x_new_indices = cupy.searchsorted(
            self._x_shift, x_new, side=self._side)

        # 2. Clip x_new_indices so that they are within the range of x indices.
        xn = len(self.x) - self._ind
        x_new_indices = x_new_indices.clip(1 - self._ind,
                                           xn).astype(cupy.intp)

        # 3. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices+self._ind-1]

        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = cupy.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x. Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = cupy.asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x. Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        if self.bounds_error and below_bounds.any():
            below_bounds_value = x_new[cupy.argmax(below_bounds)]
            raise ValueError(f"A value ({below_bounds_value}) in x_new is "
                             f"below the interpolation range's minimum value "
                             f"({self.x[0]}).")
        if self.bounds_error and above_bounds.any():
            above_bounds_value = x_new[cupy.argmax(above_bounds)]
            raise ValueError(f"A value ({above_bounds_value}) in x_new is "
                             f"above the interpolation range's maximum value "
                             f"({self.x[-1]}).")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds
