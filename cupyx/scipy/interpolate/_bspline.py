
import operator

import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename

from cupyx.scipy.sparse import csr_matrix

import numpy as np

TYPES = ['double', 'thrust::complex<double>']
INT_TYPES = ['int', 'long long']

INTERVAL_KERNEL = r'''
#include <cupy/complex.cuh>
extern "C" {
__global__ void find_interval(
        const double* t, const double* x, long long* out,
        int k, int n, bool extrapolate, int total_x) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= total_x) {
        return;
    }

    double xp = *&x[idx];
    double tb = *&t[k];
    double te = *&t[n];

    if(isnan(xp)) {
        out[idx] = -1;
        return;
    }

    if((xp < tb || xp > te) && !extrapolate) {
        out[idx] = -1;
        return;
    }

    int left = k;
    int right = n;
    int mid;
    bool found = false;

    while(left < right && !found) {
        mid = ((right + left) / 2);
        if(xp > *&t[mid]) {
            left = mid + 1;
        } else if (xp < *&t[mid]) {
            right = mid - 1;
        } else {
            found = true;
        }
    }

    int default_value = left - 1 < k ? k : left - 1;
    int result = found ? mid + 1 : default_value + 1;

    while(xp >= *&t[result] && result != n) {
        result++;
    }

    out[idx] = result - 1;
}
}
'''

INTERVAL_MODULE = cupy.RawModule(
    code=INTERVAL_KERNEL, options=('-std=c++11',),)
#    name_expressions=[f'find_interval<{type_name}>' for type_name in TYPES])


D_BOOR_KERNEL = r'''
#include <cupy/complex.cuh>
#include <cupy/math_constants.h>
#define COMPUTE_LINEAR 0x1

template<typename T>
__global__ void d_boor(
        const double* t, const T* c, const int k, const int mu,
        const double* x, const long long* intervals, T* out,
        double* temp, int num_c, int mode, int num_x) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_x) {
        return;
    }

    double xp = *&x[idx];
    long long interval = *&intervals[idx];

    double* h = temp + idx * (2 * k + 1);
    double* hh = h + k + 1;

    int ind, j, n;
    double xa, xb, w;

    if(mode == COMPUTE_LINEAR && interval < 0) {
        for(j = 0; j < num_c; j++) {
            out[num_c * idx + j] = CUDART_NAN;
        }
        return;
    }

    /*
     * Perform k-m "standard" deBoor iterations
     * so that h contains the k+1 non-zero values of beta_{ell,k-m}(x)
     * needed to calculate the remaining derivatives.
     */
    h[0] = 1.0;
    for (j = 1; j <= k - mu; j++) {
        for(int p = 0; p < j; p++) {
            hh[p] = h[p];
        }
        h[0] = 0.0;
        for (n = 1; n <= j; n++) {
            ind = interval + n;
            xb = t[ind];
            xa = t[ind - j];
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            w = hh[n - 1]/(xb - xa);
            h[n - 1] += w*(xb - xp);
            h[n] = w*(xp - xa);
        }
    }

    /*
     * Now do m "derivative" recursions
     * to convert the values of beta into the mth derivative
     */
    for (j = k - mu + 1; j <= k; j++) {
        for(int p = 0; p < j; p++) {
            hh[p] = h[p];
        }
        h[0] = 0.0;
        for (n = 1; n <= j; n++) {
            ind = interval + n;
            xb = t[ind];
            xa = t[ind - j];
            if (xb == xa) {
                h[mu] = 0.0;
                continue;
            }
            w = ((double) j) * hh[n - 1]/(xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }

    if(mode != COMPUTE_LINEAR) {
        return;
    }

    // Compute linear combinations
    for(j = 0; j < num_c; j++) {
        out[num_c * idx + j] = 0;
        for(n = 0; n < k + 1; n++) {
            out[num_c * idx + j] = (
                out[num_c * idx + j] +
                c[(interval + n - k) * num_c + j] * ((T) h[n]));
        }
    }

}
'''

D_BOOR_MODULE = cupy.RawModule(
    code=D_BOOR_KERNEL, options=('-std=c++11',),
    name_expressions=[f'd_boor<{type_name}>' for type_name in TYPES])


DESIGN_MAT_KERNEL = r'''
#include <cupy/complex.cuh>

template<typename U>
__global__ void compute_design_matrix(
        const int k, const long long* intervals, double* bspline_basis,
        double* data, U* indices, int num_intervals) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_intervals) {
        return;
    }

    long long interval = *&intervals[idx];

    double* work = bspline_basis + idx * (2 * k + 1);

    for(int j = 0; j <= k; j++) {
        int m = (k + 1) * idx + j;
        data[m] = work[j];
        indices[m] = (U) (interval - k + j);
    }
}
'''

DESIGN_MAT_MODULE = cupy.RawModule(
    code=DESIGN_MAT_KERNEL, options=('-std=c++11',),
    name_expressions=[f'compute_design_matrix<{itype}>'
                      for itype in INT_TYPES])


def _get_module_func(module, func_name, *template_args):
    def _get_typename(dtype):
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if cupy.issubdtype(dtype, cupy.complexfloating):
        return cupy.complex_
    else:
        return cupy.float_


def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.
    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = cupy.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and not cupy.isfinite(x).all():
        raise ValueError("Array must not contain infs or nans.")
    return x


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
    n = t.shape[0] - k - 1
    intervals = cupy.empty_like(xp, dtype=cupy.int64)

    # Compute intervals for each value
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                    (t, xp, intervals, k, n, extrapolate, xp.shape[0]))

    # Compute interpolation
    num_c = int(np.prod(c.shape[1:]))
    temp = cupy.empty(xp.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', c)
    d_boor_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                  (t, c, k, nu, xp, intervals, out, temp, num_c, 1,
                   xp.shape[0]))


def _make_design_matrix(x, t, k, extrapolate, indices):
    """
    Returns a design matrix in CSR format.
    Note that only indices is passed, but not indptr because indptr is already
    precomputed in the calling Python function design_matrix.

    Parameters
    ----------
    x : array_like, shape (n,)
        Points to evaluate the spline at.
    t : array_like, shape (nt,)
        Sorted 1D array of knots.
    k : int
        B-spline degree.
    extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points.
    indices : ndarray, shape (n * (k + 1),)
        Preallocated indices of the final CSR array.
    Returns
    -------
    data
        The data array of a CSR array of the b-spline design matrix.
        In each row all the basis elements are evaluated at the certain point
        (first row - x[0], ..., last row - x[-1]).

    indices
        The indices array of a CSR array of the b-spline design matrix.
    """
    n = t.shape[0] - k - 1
    intervals = cupy.empty_like(x, dtype=cupy.int64)

    # Compute intervals for each value
    interval_kernel = _get_module_func(INTERVAL_MODULE, 'find_interval')
    interval_kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                    (t, x, intervals, k, n, extrapolate, x.shape[0]))

    # Compute interpolation
    bspline_basis = cupy.empty(x.shape[0] * (2 * k + 1))
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', x)
    d_boor_kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                  (t, None, k, 0, x, intervals, None, bspline_basis, 0, 0,
                   x.shape[0]))

    data = cupy.zeros(x.shape[0] * (k + 1), dtype=cupy.float_)
    design_mat_kernel = _get_module_func(
        DESIGN_MAT_MODULE, 'compute_design_matrix', indices)
    design_mat_kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                      (k, intervals, bspline_basis, data, indices,
                       x.shape[0]))

    return data, indices


def splder(tck, n=1):
    """
    Compute the spline representation of the derivative of a given spline

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose derivative to compute
    n : int, optional
        Order of derivative to evaluate. Default: 1

    Returns
    -------
    tck_der : tuple of (t2, c2, k2)
        Spline of order k2=k-n representing the derivative
        of the input spline.

    Notes
    -----
    .. seealso:: :class:`scipy.interpolate.splder`

    See Also
    --------
    splantider, splev, spalde
    """
    if n < 0:
        return splantider(tck, -n)

    t, c, k = tck

    if n > k:
        raise ValueError(("Order of derivative (n = %r) must be <= "
                          "order of spline (k = %r)") % (n, tck[2]))

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + ((None,)*len(c.shape[1:]))

    try:
        for j in range(n):
            # See e.g. Schumaker, Spline Functions: Basic Theory, Chapter 5

            # Compute the denominator in the differentiation formula.
            # (and append traling dims, if necessary)
            dt = t[k+1:-1] - t[1:-k-1]
            dt = dt[sh]
            # Compute the new coefficients
            c = (c[1:-1-k] - c[:-2-k]) * k / dt
            # Pad coefficient array to same size as knots (FITPACK
            # convention)
            c = cupy.r_[c, np.zeros((k,) + c.shape[1:])]
            # Adjust knots
            t = t[1:-1]
            k -= 1
    except FloatingPointError as e:
        raise ValueError(("The spline has internal repeated knots "
                          "and is not differentiable %d times") % n) from e

    return t, c, k


def splantider(tck, n=1):
    """
    Compute the spline for the antiderivative (integral) of a given spline.

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline whose antiderivative to compute
    n : int, optional
        Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    tck_ader : tuple of (t2, c2, k2)
        Spline of order k2=k+n representing the antiderivative of the input
        spline.

    See Also
    --------
    splder, splev, spalde

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    .. seealso:: :class:`scipy.interpolate.splantider`
    """
    if n < 0:
        return splder(tck, -n)

    t, c, k = tck

    # Extra axes for the trailing dims of the `c` array:
    sh = (slice(None),) + (None,)*len(c.shape[1:])

    for j in range(n):
        # This is the inverse set of operations to splder.

        # Compute the multiplier in the antiderivative formula.
        dt = t[k+1:] - t[:-k-1]
        dt = dt[sh]
        # Compute the new coefficients
        c = cupy.cumsum(c[:-k-1] * dt, axis=0) / (k + 1)
        c = cupy.r_[cupy.zeros((1,) + c.shape[1:]),
                    c, [c[-1]] * (k+2)]
        # New knots
        t = cupy.r_[t[0], t, t[-1]]
        k += 1

    return t, c, k


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

    - Based on [1]_ and [2]_

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
            self.c = cupy.moveaxis(self.c, axis, 0)

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
            raise ValueError(
                "Knots, coefficients and degree are inconsistent.")

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

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        Parameters
        ----------
        t : ndarray, shape (k+2,)
            internal knots
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[0] .. t[k+1]``, or to return nans.
            If 'periodic', periodic extrapolation is used.
            Default is True.

        Returns
        -------
        basis_element : callable
            A callable representing a B-spline basis element for the knot
            vector `t`.

        Notes
        -----
        The degree of the B-spline, `k`, is inferred from the length of `t` as
        ``len(t)-2``. The knot vector is constructed by appending and
        prepending ``k+1`` elements to internal knots `t`.

        .. seealso:: :class:`scipy.interpolate.BSpline`
        """
        k = len(t) - 2
        t = _as_float_array(t)
        t = cupy.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
        c = cupy.zeros_like(t)
        c[k] = 1.
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        """
        Returns a design matrix as a CSR format sparse array.

        Parameters
        ----------
        x : array_like, shape (n,)
            Points to evaluate the spline at.
        t : array_like, shape (nt,)
            Sorted 1D array of knots.
        k : int
            B-spline degree.
        extrapolate : bool or 'periodic', optional
            Whether to extrapolate based on the first and last intervals
            or raise an error. If 'periodic', periodic extrapolation is used.
            Default is False.

        Returns
        -------
        design_matrix : `csr_matrix` object
            Sparse matrix in CSR format where each row contains all the basis
            elements of the input row (first row = basis elements of x[0],
            ..., last row = basis elements x[-1]).

        Notes
        -----
        In each row of the design matrix all the basis elements are evaluated
        at the certain point (first row - x[0], ..., last row - x[-1]).
        `nt` is a length of the vector of knots: as far as there are
        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`
        to have at least `k + 1` basis element.

        Out of bounds `x` raises a ValueError.

        .. note::
            This method returns a `csr_matrix` instance as CuPy still does not
            have `csr_array`.

        .. seealso:: :class:`scipy.interpolate.BSpline`
        """
        x = _as_float_array(x, True)
        t = _as_float_array(t, True)

        if extrapolate != 'periodic':
            extrapolate = bool(extrapolate)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError(f"Expect t to be a 1-D sorted array_like, but "
                             f"got t={t}.")
        # There are `nt - k - 1` basis elements in a BSpline built on the
        # vector of knots with length `nt`, so to have at least `k + 1` basis
        # elements we need to have at least `2 * k + 2` elements in the vector
        # of knots.
        if len(t) < 2 * k + 2:
            raise ValueError(f"Length t is not enough for k={k}.")

        if extrapolate == 'periodic':
            # With periodic extrapolation we map x to the segment
            # [t[k], t[n]].
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
        ):
            # Checks from `find_interval` function
            raise ValueError(f'Out of bounds w/ x = {x}.')

        # Compute number of non-zeros of final CSR array in order to determine
        # the dtype of indices and indptr of the CSR array.
        n = x.shape[0]
        nnz = n * (k + 1)
        if nnz < cupy.iinfo(cupy.int32).max:
            int_dtype = cupy.int32
        else:
            int_dtype = cupy.int64
        # Preallocate indptr and indices
        indices = cupy.empty(n * (k + 1), dtype=int_dtype)
        indptr = cupy.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)

        # indptr is not passed to CUDA as it is already fully computed
        data, indices = _make_design_matrix(
            x, t, k, extrapolate, indices
        )
        return csr_matrix(
            (data, indices, indptr),
            shape=(x.shape[0], t.shape[0] - k - 1)
        )

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
        x = cupy.ascontiguousarray(cupy.ravel(x), dtype=cupy.float_)

        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                         self.t[self.k])
            extrapolate = False

        out = cupy.empty(
            (len(x), int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)

        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            dim_order = list(range(out.ndim))
            dim_order = (
                dim_order[x_ndim:x_ndim+self.axis] +
                dim_order[:x_ndim] +
                dim_order[x_ndim+self.axis:])
            out = out.transpose(dim_order)

        return out

    def _ensure_c_contiguous(self):
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def _evaluate(self, xp, nu, extrapolate, out):
        _evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
                         self.k, xp, nu, extrapolate, out)

    def derivative(self, nu=1):
        """
        Return a B-spline representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Derivative order.
            Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the derivative.

        See Also
        --------
        splder, splantider
        """
        c = self.c
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splder((self.t, c, self.k), nu)
        return self.construct_fast(*tck, extrapolate=self.extrapolate,
                                   axis=self.axis)

    def antiderivative(self, nu=1):
        """
        Return a B-spline representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the antiderivative.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.

        See Also
        --------
        splder, splantider
        """
        c = self.c
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        tck = splantider((self.t, c, self.k), nu)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(*tck, extrapolate=extrapolate,
                                   axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral of the spline.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
            base interval. If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        I : array_like
            Definite integral of the spline over the interval ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Prepare self.t and self.c.
        self._ensure_c_contiguous()

        # Swap integration bounds if needed.
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        n = self.t.size - self.k - 1

        if extrapolate != "periodic" and not extrapolate:
            # Shrink the integration interval, if needed.
            a = max(a, self.t[self.k].item())
            b = min(b, self.t[n].item())

            # if self.c.ndim == 1:
            #     # Fast path: use FITPACK's routine
            #     # (cf _fitpack_impl.splint).
            #     integral = splint(a, b, self.tck)
            #     return integral * sign

        out = cupy.empty(
            (2, int(np.prod(self.c.shape[1:]))), dtype=self.c.dtype)

        # Compute the antiderivative.
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = cupy.r_[c, cupy.zeros((ct,) + c.shape[1:])]
        ta, ca, ka = splantider((self.t, c, self.k), 1)

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            ts, te = self.t[self.k], self.t[n]
            period = te - ts
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                # Evaluate the difference of antiderivatives.
                x = cupy.asarray([ts, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                 ka, x, 0, False, out)
                integral = out[1] - out[0]
                integral *= n_periods
            else:
                integral = cupy.zeros((1, int(np.prod(self.c.shape[1:]))),
                                      dtype=self.c.dtype)

            # Map a to [ts, te], b is always a + left.
            a = ts + (a - ts) % period
            b = a + left

            # If b <= te then we need to integrate over [a, b], otherwise
            # over [a, te] and from xs to what is remained.
            if b <= te:
                x = cupy.asarray([a, b], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                 ka, x, 0, False, out)
                integral += out[1] - out[0]
            else:
                x = cupy.asarray([a, te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                 ka, x, 0, False, out)
                integral += out[1] - out[0]

                x = cupy.asarray([ts, ts + b - te], dtype=cupy.float_)
                _evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                 ka, x, 0, False, out)
                integral += out[1] - out[0]
        else:
            # Evaluate the difference of antiderivatives.
            x = cupy.asarray([a, b], dtype=cupy.float_)
            _evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                             ka, x, 0, extrapolate, out)
            integral = out[1] - out[0]

        integral *= sign
        return integral.reshape(ca.shape[1:])
