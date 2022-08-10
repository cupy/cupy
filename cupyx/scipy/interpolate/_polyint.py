import cupy
from cupyx.scipy._lib._util import _asarray_validated


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

        Parametres
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
            self.dtype = cupy.complex_
        else:
            if not union or self.dtype != cupy.complex_:
                self.dtype = cupy.float_


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

        self.xi = xi.astype(cupy.float_)
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

        return _Interpolator1D.__call__(self, x)

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