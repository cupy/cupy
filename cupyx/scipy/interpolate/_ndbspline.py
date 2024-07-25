import itertools
import operator
from math import prod

import cupy

from cupyx.scipy.interpolate._bspline import _get_dtype, _get_module_func
from cupyx.scipy.interpolate._bspline2 import _not_a_knot
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve


TYPES = ['double', 'thrust::complex<double>']

NDBSPL_DEF = r"""
#include <cupy/complex.cuh>
#include <cupy/math_constants.h>

__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock   = blockDim.x;
    const int curThreadIdx    = ( blockIdx.x * threadsPerBlock ) + threadIdx.x;
    return curThreadIdx;
}
__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid     = gridDim.x;
    const int threadsPerBlock   = blockDim.x;
    const int threadNum         = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

__device__ long long find_interval(
        const double* t, double xp, int k, int n, bool extrapolate) {

    double tb = *&t[k];
    double te = *&t[n];

    if(isnan(xp)) {
        return -1;
    }

    if((xp < tb || xp > te) && !extrapolate) {
        return -1;
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

    return result - 1;
}

__device__ void d_boor(
        const double* t, double xp, long long interval, const long long k,
        const int mu, double* temp) {

    double* h = temp;
    double* hh = h + k + 1;

    int ind, j, n;
    double xa, xb, w;

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

}

__global__ void compute_nd_bsplines(
        const double* xi, int n_xi, const double* t, const long long* t_sz,
        int ndim, int max_t, const long long* k, const long long* max_k,
        const int* nu, bool extrapolate, bool check_all_validity,
        long long* intervals, double* splines, bool* invalid) {

    int total = n_xi * ndim;

    for(int midx = getCurThreadIdx(); midx < total; midx += getThreadNum()) {
        int idx = midx / ndim;
        int dim_idx = midx % ndim;

        double xd = xi[ndim * idx + dim_idx];
        const double* dim_t = t + max_t * dim_idx;
        const long long dim_k = k[dim_idx];
        const long long dim_t_sz = t_sz[dim_idx];

        long long interval = find_interval(
            dim_t, xd, dim_k, dim_t_sz - dim_k - 1, extrapolate);

        if(interval < 0) {
            invalid[check_all_validity ? idx : blockIdx.x] = true;
            continue;
        }

        intervals[ndim * idx + dim_idx] = interval;
        double* dim_splines = (
            splines + ndim * (2 * max_k[0] + 2) * idx +
            (2 * max_k[0] + 2) * dim_idx);

        d_boor(dim_t, xd, interval, dim_k, nu[dim_idx], dim_splines);
    }
}

template<typename T>
__global__ void eval_nd_bspline(
        const long long* indices_k1d, const long long* strides_c1,
        const double* b, const long long* intervals, const long long* k,
        bool* invalid, T* c1r, long long* volume, int ndim, int num_c,
        int n_xi, const long long* max_k, T* out) {

    for(int idx = getCurThreadIdx(); idx < n_xi; idx += getThreadNum()) {
        if(invalid[idx]) {
            for(int i = 0; i < num_c; i++) {
                out[num_c * idx + i] = CUDART_NAN;
            }
            continue;
        }

        for(int i = 0; i < num_c; i++) {
            out[num_c * idx + i] = 0;
        }

        const double* idx_splines = b + ndim * (2 * max_k[0] + 2) * idx;

        for(long long iflat = 0; iflat < *volume; iflat++) {
            const long long* idx_b = indices_k1d + ndim * iflat;
            long long idx_cflat_base = 0;
            double factor = 1.0;

            for(int d = 0; d < ndim; d++) {
                const double* dim_splines = (
                    idx_splines + (2 * max_k[0] + 2) * d);
                factor *= dim_splines[idx_b[d]];
                long long d_idx = idx_b[d] + intervals[ndim * idx + d] - k[d];
                idx_cflat_base += d_idx * strides_c1[d];
            }

            for(int i = 0; i < num_c; i++) {
                out[num_c * idx + i] += c1r[idx_cflat_base + i] * factor;
            }
        }
    }
}


__global__ void store_nd_bsplines(
        const long long* indices_k1d, const long long* strides_c1,
        const double* b, const long long* intervals, const long long* k,
        long long* volume, int ndim, int n_xi, const long long* max_k,
        long long* out_idx, double* out) {

    int total = n_xi * volume[0];

    for(int midx = getCurThreadIdx(); midx < total; midx += getThreadNum()) {
        int idx = midx / volume[0];
        int iflat = midx % volume[0];

        const double* idx_splines = b + ndim * (2 * max_k[0] + 2) * idx;
        const long long* idx_b = indices_k1d + ndim * iflat;

        long long idx_cflat_base = 0;
        double factor = 1.0;

        for(int d = 0; d < ndim; d++) {
            const double* dim_splines = (
                idx_splines + (2 * max_k[0] + 2) * d);
            factor *= dim_splines[idx_b[d]];
            long long d_idx = idx_b[d] + intervals[ndim * idx + d] - k[d];
            idx_cflat_base += d_idx * strides_c1[d];
        }

        out_idx[volume[0] * idx + iflat] = idx_cflat_base;
        out[volume[0] * idx + iflat] = factor;
    }
}
"""

NDBSPL_MOD = cupy.RawModule(
    code=NDBSPL_DEF, options=('-std=c++11',),
    name_expressions=['compute_nd_bsplines', 'store_nd_bsplines'] +
                     [f'eval_nd_bspline<{t}>' for t in TYPES])


def evaluate_ndbspline(
        xi, t, len_t, k, nu, extrapolate, c1r, num_c_tr,
        strides_c1, indices_k1d, out):
    """Evaluate an N-dim tensor product spline or its derivative.

    Parameters
    ----------
    xi : ndarray, shape(npoints, ndim)
        ``npoints`` values to evaluate the spline at, each value is
        a point in an ``ndim``-dimensional space.
    t : ndarray, shape(ndim, max_len_t)
        Array of knots for each dimension.
        This array packs the tuple of knot arrays per dimension into a single
        2D array. The array is ragged (knot lengths may differ), hence
        the real knots in dimension ``d`` are ``t[d, :len_t[d]]``.
    len_t : ndarray, 1D, shape (ndim,)
        Lengths of the knot arrays, per dimension.
    k : tuple of ints, len(ndim)
        Spline degrees in each dimension.
    nu : ndarray of ints, shape(ndim,)
        Orders of derivatives to compute, per dimension.
    extrapolate : int
        Whether to extrapolate out of bounds or return nans.
    c1r: ndarray, one-dimensional
        Flattened array of coefficients.
        The original N-dimensional coefficient array ``c`` has shape
        ``(n1, ..., nd, ...)`` where each ``ni == len(t[d]) - k[d] - 1``,
        and the second "..." represents trailing dimensions of ``c``.
        In code, given the C-ordered array ``c``, ``c1r`` is
        ``c1 = c.reshape(c.shape[:ndim] + (-1,)); c1r = c1.ravel()``
    num_c_tr : int
        The number of elements of ``c1r``, which correspond to the trailing
        dimensions of ``c``. In code, this is
        ``c1 = c.reshape(c.shape[:ndim] + (-1,)); num_c_tr = c1.shape[-1]``.
    strides_c1 : ndarray, one-dimensional
        Pre-computed strides of the ``c1`` array.
        Note: These are *data* strides, not numpy-style byte strides.
        This array is equivalent to
        ``[stride // s1.dtype.itemsize for stride in s1.strides]``.
    indices_k1d : ndarray, shape((k+1)**ndim, ndim)
        Pre-computed mapping between indices for iterating over a flattened
        array of shape ``[k[d] + 1) for d in range(ndim)`` and
        ndim-dimensional indices of the ``(k+1,)*ndim`` dimensional array.
        This is essentially a transposed version of
        ``cupy.unravel_index(cupy.arange((k+1)**ndim), (k+1,)*ndim)``.
    out : ndarray, shape (npoints, num_c_tr)
        Output values of the b-spline at given ``xi`` points.

    Notes
    -----

    This function is essentially equivalent to the following: given an
    N-dimensional vector ``x = (x1, x2, ..., xN)``, iterate over the
    dimensions, form linear combinations of products,
    B(x1) * B(x2) * ... B(xN) of (k+1)**N b-splines which are non-zero
    at ``x``.

    Since b-splines are localized, the sum has (k+1)**N non-zero elements.

    If ``i = (i1, i2, ..., iN)`` is a vector if intervals of the knot
    vectors, ``t[d, id] <= xd < t[d, id+1]``, for ``d=1, 2, ..., N``, then
    the core loop of this function is nothing but

    ```
    result = 0
    iters = [range(i[d] - self.k[d], i[d] + 1) for d in range(ndim)]
    for idx in itertools.product(*iters):
        term = self.c[idx] * cupy.prod([B(x[d], self.k[d], idx[d], self.t[d])
                                        for d in range(ndim)])
        result += term
    ```

    For efficiency reasons, we iterate over the flattened versions of the
    arrays.

    """
    max_k = k.max()
    volume = cupy.prod(k + 1)
    intervals = cupy.empty((xi.shape[0], t.shape[0]), dtype=cupy.int64)
    splines = cupy.empty((xi.shape[0], t.shape[0], 2 * max_k.item() + 2),
                         dtype=cupy.float64)
    invalid = cupy.zeros(xi.shape[0], dtype=cupy.bool_)

    compute_nd_bsplines = NDBSPL_MOD.get_function('compute_nd_bsplines')
    compute_nd_bsplines((512,), (128,), (
        xi, xi.shape[0], t, len_t, xi.shape[1], t.shape[1], k, max_k,
        nu, extrapolate, True, intervals, splines, invalid
    ))

    eval_nd_bspline = _get_module_func(NDBSPL_MOD, 'eval_nd_bspline', c1r)
    eval_nd_bspline((512,), (128,), (
        indices_k1d, strides_c1, splines, intervals, k, invalid, c1r, volume,
        xi.shape[1], num_c_tr, xi.shape[0], max_k, out))


def colloc_nd(xvals, t, len_t, k):
    """Construct the N-D tensor product collocation matrix as a CSR array.

    In the dense representation, each row of the collocation matrix corresponds
    to a data point and contains non-zero b-spline basis functions which are
    non-zero at this data point.

    Parameters
    ----------
    xvals : ndarray, shape(size, ndim)
        Data points. ``xvals[j, :]`` gives the ``j``-th data point as an
        ``ndim``-dimensional array.
    t : tuple of 1D arrays, length-ndim
        Tuple of knot vectors
    k : ndarray, shape (ndim,)
        Spline degrees

    Returns
    -------
    csr_data, csr_indices, csr_indptr
        The collocation matrix in the CSR array format.

    Notes
    -----
    Algorithm: given `xvals` and the tuple of knots `t`, we construct a tensor
    product spline, i.e. a linear combination of

        B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)


    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Since ``B`` functions are localized, for each point `(x1, ..., xN)` we
    loop over the dimensions, and
    - find the the location in the knot array, `t[i] <= x < t[i+1]`,
    - compute all non-zero `B` values
    - place these values into the relevant row

    In the dense representation, the collocation matrix would have had a row
    per data point, and each row has the values of the basis elements
    (i.e., tensor products of B-splines) evaluated at this data point.
    Since the matrix is very sparse (has size = len(x)**ndim, with only
    (k+1)**ndim non-zero elements per row), we construct it in the CSR format.
    """
    size = xvals.shape[0]
    ndim = xvals.shape[1]

    max_k = k.max()
    volume = cupy.prod(k + 1)
    cpu_volume = volume.get()

    intervals = cupy.empty((xvals.shape[0], t.shape[0]), dtype=cupy.int64)
    splines = cupy.empty((xvals.shape[0], t.shape[0], 2 * max_k.item() + 2),
                         dtype=cupy.float64)
    invalid = cupy.zeros(512, dtype=cupy.bool_)
    nu = cupy.zeros(ndim, dtype=cupy.int64)

    k1_shape = tuple(kd + 1 for kd in k.get())

    # Precompute the shape and strides of the coefficients array.
    # This would have been the NdBSpline coefficients; in the present context
    # this is a helper to compute the indices into the collocation matrix.
    c_shape = len_t - cupy.asarray(k1_shape, dtype=len_t.dtype)

    # The computation is equivalent to
    # >>> x = cupy.empty(c_shape)
    # >>> cstrides = [s // 8 for s in x.strides]
    cs = cupy.r_[c_shape[1:], 1]
    cstrides = cupy.cumprod(cs[::-1], dtype=cupy.int64)[::-1].copy()

    # tabulate flat indices for iterating over the (k+1)**ndim subarray of
    # non-zero b-spline elements
    indices = cupy.unravel_index(cupy.arange(cpu_volume), k1_shape)
    _indices_k1d = cupy.asarray(indices, dtype=cupy.int64).T.copy()

    # Allocate the collocation matrix in the CSR format.
    # If dense, this would have been
    # >>> matr = cupy.zeros((size, max_row_index), dtype=float)
    csr_indices = cupy.empty(shape=(size * cpu_volume,), dtype=cupy.int64)
    csr_data = cupy.empty(shape=(size * cpu_volume,), dtype=cupy.float64)
    csr_indptr = cupy.arange(
        0, cpu_volume * size + 1, cpu_volume, dtype=cupy.int64)

    compute_nd_bsplines = NDBSPL_MOD.get_function('compute_nd_bsplines')
    compute_nd_bsplines((512,), (128,), (
        xvals, xvals.shape[0], t, len_t, xvals.shape[1], t.shape[1], k, max_k,
        nu, True, False, intervals, splines, invalid
    ))

    if cupy.any(invalid).item():
        raise ValueError('Out of bounds')

    store_nd_splines = NDBSPL_MOD.get_function('store_nd_bsplines')
    store_nd_splines((512,), (128,), (
        _indices_k1d, cstrides, splines, intervals, k, volume, int(ndim),
        int(size), max_k, csr_indices, csr_data
    ))
    return csr_data, csr_indices, csr_indptr


class NdBSpline:
    """Tensor product spline object.

    The value at point ``xp = (x1, x2, ..., xN)`` is evaluated as a linear
    combination of products of one-dimensional b-splines in each of the ``N``
    dimensions::

       c[i1, i2, ..., iN] * B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)


    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Parameters
    ----------
    t : tuple of 1D ndarrays
        knot vectors in directions 1, 2, ... N,
        ``len(t[i]) == n[i] + k + 1``
    c : ndarray, shape (n1, n2, ..., nN, ...)
        b-spline coefficients
    k : int or length-d tuple of integers
        spline degrees.
        A single integer is interpreted as having this degree for
        all dimensions.
    extrapolate : bool, optional
        Whether to extrapolate out-of-bounds inputs, or return `nan`.
        Default is to extrapolate.

    Attributes
    ----------
    t : tuple of ndarrays
        Knots vectors.
    c : ndarray
        Coefficients of the tensor-produce spline.
    k : tuple of integers
        Degrees for each dimension.
    extrapolate : bool, optional
        Whether to extrapolate or return nans for out-of-bounds inputs.
        Defaults to true.

    See Also
    --------
    BSpline : a one-dimensional B-spline object
    NdPPoly : an N-dimensional piecewise tensor product polynomial

    """

    def __init__(self, t, c, k, *, extrapolate=None):
        ndim = len(t)

        try:
            len(k)
        except TypeError:
            # make k a tuple
            k = (k,) * ndim

        if len(k) != ndim:
            raise ValueError(f"{len(t) = } != {len(k) = }.")

        self.k = tuple(operator.index(ki) for ki in k)
        self.t = tuple(cupy.ascontiguousarray(ti, dtype=float) for ti in t)
        self.c = cupy.asarray(c)

        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        self.c = cupy.asarray(c)

        for d in range(ndim):
            td = self.t[d]
            kd = self.k[d]
            n = td.shape[0] - kd - 1
            if kd < 0:
                raise ValueError(f"Spline degree in dimension {d} cannot be"
                                 f" negative.")
            if td.ndim != 1:
                raise ValueError(f"Knot vector in dimension {d} must be"
                                 f" one-dimensional.")
            if n < kd + 1:
                raise ValueError(f"Need at least {2*kd + 2} knots for degree"
                                 f" {kd} in dimension {d}.")
            if (cupy.diff(td) < 0).any():
                raise ValueError(f"Knots in dimension {d} must be in a"
                                 f" non-decreasing order.")
            if len(cupy.unique(td[kd:n + 1])) < 2:
                raise ValueError(f"Need at least two internal knots in"
                                 f" dimension {d}.")
            if not cupy.isfinite(td).all():
                raise ValueError(f"Knots in dimension {d} should not have"
                                 f" nans or infs.")
            if self.c.ndim < ndim:
                raise ValueError(f"Coefficients must be at least"
                                 f" {d}-dimensional.")
            if self.c.shape[d] != n:
                raise ValueError(f"Knots, coefficients and degree in dimension"
                                 f" {d} are inconsistent:"
                                 f" got {self.c.shape[d]} coefficients for"
                                 f" {len(td)} knots, need at least {n} for"
                                 f" k={k}.")

        dt = _get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dt)

    def __call__(self, xi, *, nu=None, extrapolate=None):
        """Evaluate the tensor product b-spline at ``xi``.

        Parameters
        ----------
        xi : array_like, shape(..., ndim)
            The coordinates to evaluate the interpolator at.
            This can be a list or tuple of ndim-dimensional points
            or an array with the shape (num_points, ndim).
        nu : array_like, optional, shape (ndim,)
            Orders of derivatives to evaluate. Each must be non-negative.
            Defaults to the zeroth derivivative.
        extrapolate : bool, optional
            Whether to exrapolate based on first and last intervals in each
            dimension, or return `nan`. Default is to ``self.extrapolate``.

        Returns
        -------
        values : ndarray, shape ``xi.shape[:-1] + self.c.shape[ndim:]``
            Interpolated values at ``xi``
        """
        ndim = len(self.t)

        if extrapolate is None:
            extrapolate = self.extrapolate
        extrapolate = bool(extrapolate)

        if nu is None:
            nu = cupy.zeros((ndim,), dtype=cupy.int32)
        else:
            nu = cupy.asarray(nu, dtype=cupy.int32)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError(
                    f"invalid number of derivative orders {nu = } for "
                    f"ndim = {len(self.t)}.")
            if cupy.any(nu < 0).item():
                raise ValueError(f"derivatives must be positive, got {nu = }")

        # prepare xi : shape (..., m1, ..., md) -> (1, m1, ..., md)
        xi = cupy.asarray(xi, dtype=float)
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = cupy.ascontiguousarray(xi)

        if xi_shape[-1] != ndim:
            raise ValueError(f"Shapes: xi.shape={xi_shape} and ndim={ndim}")

        # prepare k & t
        _k = cupy.asarray(self.k, dtype=cupy.int64)

        # pack the knots into a single array
        len_t = [len(ti) for ti in self.t]
        _t = cupy.empty((ndim, max(len_t)), dtype=float)
        _t.fill(cupy.nan)
        for d in range(ndim):
            _t[d, :len(self.t[d])] = self.t[d]
        len_t = cupy.asarray(len_t, dtype=cupy.int64)

        # tabulate the flat indices for iterating over the (k+1)**ndim subarray
        shape = tuple(kd + 1 for kd in self.k)
        indices = cupy.unravel_index(cupy.arange(prod(shape)), shape)
        _indices_k1d = cupy.asarray(indices, dtype=cupy.int64).T.copy()

        # prepare the coefficients: flatten the trailing dimensions
        c1 = self.c.reshape(self.c.shape[:ndim] + (-1,))
        c1r = c1.ravel()

        # replacement for cupy.ravel_multi_index for indexing of `c1`:
        _strides_c1 = cupy.asarray([s // c1.dtype.itemsize
                                    for s in c1.strides], dtype=cupy.int64)

        num_c_tr = c1.shape[-1]  # # of trailing coefficients
        out = cupy.empty(xi.shape[:-1] + (num_c_tr,), dtype=c1.dtype)

        evaluate_ndbspline(xi,
                           _t,
                           len_t,
                           _k,
                           nu,
                           extrapolate,
                           c1r,
                           num_c_tr,
                           _strides_c1,
                           _indices_k1d,
                           out,)

        return out.reshape(xi_shape[:-1] + self.c.shape[ndim:])

    @classmethod
    def design_matrix(cls, xvals, t, k, extrapolate=True):
        """Construct the design matrix as a CSR format sparse array.

        Parameters
        ----------
        xvals :  ndarray, shape(npts, ndim)
            Data points. ``xvals[j, :]`` gives the ``j``-th data point as an
            ``ndim``-dimensional array.
        t : tuple of 1D ndarrays, length-ndim
            Knot vectors in directions 1, 2, ... ndim,
        k : int
            B-spline degree.
        extrapolate : bool, optional
            Whether to extrapolate out-of-bounds values of raise a `ValueError`

        Returns
        -------
        design_matrix : a CSR matrix
            Each row of the design matrix corresponds to a value in `xvals` and
            contains values of b-spline basis elements which are non-zero
            at this value.

        """
        xvals = cupy.asarray(xvals, dtype=cupy.float64)
        ndim = xvals.shape[-1]
        if len(t) != ndim:
            raise ValueError(
                f"Data and knots are inconsistent: len(t) = {len(t)} for "
                f" {ndim = }."
            )
        try:
            len(k)
        except TypeError:
            # make k a tuple
            k = (k,)*ndim

        len_t = [len(ti) for ti in t]
        _t = cupy.empty((ndim, max(len_t)), dtype=float)
        _t.fill(cupy.nan)
        for d in range(ndim):
            _t[d, :len(t[d])] = t[d]
        len_t = cupy.asarray(len_t, dtype=cupy.int64)

        kk = cupy.asarray(k, dtype=cupy.int64)
        data, indices, indptr = colloc_nd(xvals, _t, len_t, kk)
        return csr_matrix((data, indices, indptr))


def make_ndbspl(points, values, k=3):
    """Construct an interpolating NdBspline.

    Parameters
    ----------
    points : tuple of ndarrays of float, with shapes (m1,), ... (mN,)
        The points defining the regular grid in N dimensions. The points in
        each dimension (i.e. every element of the `points` tuple) must be
        strictly ascending or descending.
    values : ndarray of float, shape (m1, ..., mN, ...)
        The data on the regular grid in n dimensions.
    k : int, optional
        The spline degree. Must be odd. Default is cubic, k=3
    solver : a `scipy.sparse.linalg` solver (iterative or direct), optional.
        An iterative solver from `scipy.sparse.linalg` or a direct one,
        `sparse.sparse.linalg.spsolve`.
        Used to solve the sparse linear system
        ``design_matrix @ coefficients = rhs`` for the coefficients.
        Default is `scipy.sparse.linalg.gcrotmk`
    solver_args : dict, optional
        Additional arguments for the solver. The call signature is
        ``solver(csr_array, rhs_vector, **solver_args)``

    Returns
    -------
    spl : NdBSpline object

    Notes
    -----
    Boundary conditions are not-a-knot in all dimensions.
    """
    ndim = len(points)
    xi_shape = tuple(len(x) for x in points)

    try:
        len(k)
    except TypeError:
        # make k a tuple
        k = (k,)*ndim

    for d, point in enumerate(points):
        numpts = len(cupy.atleast_1d(point))
        if numpts <= k[d]:
            raise ValueError(f"There are {numpts} points in dimension {d},"
                             f" but order {k[d]} requires at least "
                             f" {k[d]+1} points per dimension.")

    t = tuple(_not_a_knot(cupy.asarray(
        points[d], dtype=float), k[d]) for d in range(ndim))
    xvals = cupy.asarray(
        [xv for xv in itertools.product(*points)], dtype=float)

    # construct the colocation matrix
    matr = NdBSpline.design_matrix(xvals, t, k)

    # Solve for the coefficients given `values`.
    # Trailing dimensions: first ndim dimensions are data, the rest are batch
    # dimensions, so stack `values` into a 2D array for `spsolve` to
    # understand.
    v_shape = values.shape
    vals_shape = (prod(v_shape[:ndim]), prod(v_shape[ndim:]))
    vals = values.reshape(vals_shape)

    if cupy.issubdtype(vals.dtype, cupy.complexfloating):
        # avoid upcasting the l.h.s. to complex (that doubles the memory)
        coef = (spsolve(matr, vals.real) +
                spsolve(matr, vals.imag) * 1.j)
    else:
        coef = spsolve(matr, vals)
    coef = coef.reshape(xi_shape + v_shape[ndim:])
    return NdBSpline(t, coef, k)
