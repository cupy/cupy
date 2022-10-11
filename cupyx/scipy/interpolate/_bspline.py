
import operator

import cupy
from cupy._core import internal

import numpy as np


INTERVAL_KERNEL = cupy.RawKernel(r'''
extern "C" __global__
void find_interval(
        const double* t, const double* x, long long* out,
        int k, int n, bool extrapolate) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float xp = x[idx];
    float tb = t[k];
    float te = t[n];

    if(isnan(xp)) {
        out[idx] = -1;
        return;
    }

    if((xp > tb || xp < te) && !extrapolate) {
        out[idx] = -1;
        return;
    }

    int left = k;
    int right = n;
    int mid = ((right - left) / 2) + k;

    while(mid != left) {
        mid = ((right - left) / 2) + k;
        if(xp > t[mid]) {
            left = mid;
        } else {
            right = mid;
        }
    }

    out[idx] = mid;
}
''', 'find_interval')


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if cupy.issubdtype(dtype, cupy.complexfloating):
        return cupy.complex_
    else:
        return cupy.float_


def _evaluate_spline(t, c, k, xp, nu, extrapolate, out):
    """
    Evaluate a spline in the B-spline basis.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        knots
    c : ndarray, shape (n, m)
        B-spline coefficients
    xp : ndarray, shape (s,)
        Points to evaluate the spline at.
    nu : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    out : ndarray, shape (s, m)
        Computed values of the spline at each of the input points.
        This argument is modified in-place.
    """




class BSpline:
    r"""Univariate spline in the B-spline basis.

    .. math::
        S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    extrapolate : bool
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
    axis : int
        Interpolation axis.
    tck : tuple
        A read-only equivalent of ``(self.t, self.c, self.k)``

    Methods
    -------
    __call__
    basis_element
    derivative
    antiderivative
    integrate
    construct_fast
    design_matrix
    from_power_basis

    Notes
    -----
    B-spline basis elements are defined via

    .. math::
        B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **Implementation details**

    - At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.

    - B-spline basis elements of degree `k` form a partition of unity on the
      *base interval*, ``t[k] <= x <= t[n]``.

    .. seealso:: :class:`scipy.interpolate.BSpline`

    References
    ----------
    .. [1] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.
    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        self.k = operator.index(k)
        self.c = cupy.asarray(c)
        self.t = cupy.ascontiguousarray(t, dtype=cupy.float64)

        if extrapolate == 'periodic':
            self.extrapolate = extrapolate
        else:
            self.extrapolate = bool(extrapolate)

        n = self.t.shape[0] - self.k - 1

        axis = internal._normalize_axis_index(axis, self.c.ndim)

        # Note that the normalized axis is stored in the object.
        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (n, ...),
            # and axis !=0 means that we have c.shape (..., n, ...)
            #                                               ^
            #                                              axis
            self.c = np.moveaxis(self.c, axis, 0)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if n < self.k + 1:
            raise ValueError("Need at least %d knots for degree %d" %
                    (2*k + 2, k))
        if (cupy.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if len(cupy.unique(self.t[k:n+1])) < 2:
            raise ValueError("Need at least two internal knots.")
        if not cupy.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        if self.c.ndim < 1:
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if self.c.shape[0] < n:
            raise ValueError("Knots, coefficients and degree are inconsistent.")

        dt = _get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dt)

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.c, self.k = t, c, k
        self.extrapolate = extrapolate
        self.axis = axis
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
        return self.t, self.c, self.k

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu : int, optional
            derivative to evaluate (default is 0).
        extrapolate : bool or 'periodic', optional
            whether to extrapolate based on the first and last intervals
            or return nans. If 'periodic', periodic extrapolation is used.
            Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        x = cupy.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = cupy.ascontiguousarray(cupy.ravel(), dtype=cupy.float_)

        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                         self.t[self.k])
            extrapolate = False

        out = cupy.empty(
            (len(x), np.prod(self.c.shape[1:])), dtype=self.c.dtype)

        self._evaluate(x, nu, extrapolate, out)


    def _evaluate(self, xp, nu, extrapolate, out):
        _evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
                         self.k, xp, nu, extrapolate, out)
