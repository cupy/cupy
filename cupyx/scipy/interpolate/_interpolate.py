
import math

import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype

import numpy as np

try:
    from math import comb
    _comb = comb
except ImportError:
    def _comb(n, k):
        return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))


TYPES = ['double', 'thrust::complex<double>']
INT_TYPES = ['int', 'long long']

INTERVAL_KERNEL = r'''
#include <cupy/complex.cuh>

#define le_or_ge(x, y, r) ((r) ? ((x) < (y)) : ((x) > (y)))
#define ge_or_le(x, y, r) ((r) ? ((x) > (y)) : ((x) < (y)))
#define geq_or_leq(x, y, r) ((r) ? ((x) >= (y)) : ((x) <= (y)))

extern "C" {
__global__ void find_breakpoint_position(
        const double* breakpoints, const double* x, long long* out,
        bool extrapolate, int total_x, int total_breakpoints,
        const bool* pasc) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= total_x) {
        return;
    }

    double xp = *&x[idx];
    double a = *&breakpoints[0];
    double b = *&breakpoints[total_breakpoints - 1];
    bool asc = pasc[0];

    if(isnan(xp)) {
        out[idx] = -1;
        return;
    }

    if(le_or_ge(xp, a, asc) || ge_or_le(xp, b, asc)) {
        if(!extrapolate) {
            out[idx] = -1;
        } else if(le_or_ge(xp, a, asc)) {
            out[idx] = 0;
        } else if(ge_or_le(xp, b, asc)) {
            out[idx] = total_breakpoints - 2;
        }
        return;
    } else if (xp == b) {
        out[idx] = total_breakpoints - 2;
        return;
    }

    int left = 0;
    int right = total_breakpoints - 2;
    int mid;

    if(le_or_ge(xp, *&breakpoints[left + 1], asc)) {
        right = left;
    }

    bool found = false;

    while(left < right && !found) {
        mid = ((right + left) / 2);
        if(le_or_ge(xp, *&breakpoints[mid], asc)) {
            right = mid;
        } else if (geq_or_leq(xp, *&breakpoints[mid + 1], asc)) {
            left = mid + 1;
        } else {
            found = true;
            left = mid;
        }
    }

    out[idx] = left;
}
}
'''

INTERVAL_MODULE = cupy.RawModule(
    code=INTERVAL_KERNEL, options=('-std=c++11',),)

if runtime.is_hip:
    BASE_HEADERS = """#include <hip/hip_runtime.h>
"""
else:
    BASE_HEADERS = """#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

PPOLY_KERNEL = BASE_HEADERS + r"""
#include <cupy/complex.cuh>
#include <cupy/math_constants.h>

template<typename T>
__device__ T eval_poly_1(
        const double s, const T* coef, long long ci, int cj, int dx,
        const long long* c_dims, const long long stride_0,
        const long long stride_1) {
    int kp, k;
    T res, z;
    double prefactor;

    res = 0.0;
    z = 1.0;

    if(dx < 0) {
        for(int i = 0; i < -dx; i++) {
            z *= s;
        }
    }

    int c_dim_0 = (int) *&c_dims[0];

    for(kp = 0; kp < c_dim_0; kp++) {
        if(dx == 0) {
            prefactor = 1.0;
        } else if(dx > 0) {
            if(kp < dx) {
                continue;
            } else {
                prefactor = 1.0;
                for(k = kp; k > kp - dx; k--) {
                    prefactor *= k;
                }
            }
        } else {
            prefactor = 1.0;
            for(k = kp; k < kp - dx; k++) {
                prefactor /= k + 1;
            }
        }

        int off = stride_0 * (c_dim_0 - kp - 1) + stride_1 * ci + cj;
        T cur_coef = *&coef[off];
        res += cur_coef * z * ((T) prefactor);

        if((kp < c_dim_0 - 1) && kp >= dx) {
            z *= s;
        }

    }

    return res;

}

template<typename T>
__global__ void eval_ppoly(
        const T* coef, const double* breakpoints, const double* x,
        const long long* intervals, int dx, const long long* c_dims,
        const long long* c_strides, int num_x, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_x) {
        return;
    }

    double xp = *&x[idx];
    long long interval = *&intervals[idx];
    double breakpoint = *&breakpoints[interval];

    const int num_c = *&c_dims[2];
    const long long stride_0 = *&c_strides[0];
    const long long stride_1 = *&c_strides[1];

    if(interval < 0) {
        for(int j = 0; j < num_c; j++) {
            out[num_c * idx + j] = CUDART_NAN;
        }
        return;
    }

    for(int j = 0; j < num_c; j++) {
        T res = eval_poly_1<T>(
            xp - breakpoint, coef, interval, ((long long) (j)), dx,
            c_dims, stride_0, stride_1);
        out[num_c * idx + j] = res;
    }
}

template<typename T>
__global__ void fix_continuity(
        T* coef, const double* breakpoints, const int order,
        const long long* c_dims, const long long* c_strides,
        int num_breakpoints) {

    const long long c_size0 = *&c_dims[0];
    const long long c_size2 = *&c_dims[2];
    const long long stride_0 = *&c_strides[0];
    const long long stride_1 = *&c_strides[1];
    const long long stride_2 = *&c_strides[2];

    for(int idx = 1; idx < num_breakpoints - 1; idx++) {
        const double breakpoint = *&breakpoints[idx];
        const long long interval = idx - 1;
        const double breakpoint_interval = *&breakpoints[interval];

        for(int jp = 0; jp < c_size2; jp++) {
            for(int dx = order; dx > -1; dx--) {
                T res = eval_poly_1<T>(
                    breakpoint - breakpoint_interval, coef,
                    interval, jp, dx, c_dims, stride_0, stride_1);

                for(int kp = 0; kp < dx; kp++) {
                    res /= kp + 1;
                }

                const long long c_idx = (
                    stride_0 * (c_size0 - dx - 1) + stride_1 * idx +
                    stride_2 * jp);

                coef[c_idx] = res;
            }
        }
    }
}

template<typename T>
__global__ void integrate(
        const T* coef, const double* breakpoints,
        const double* a_val, const double* b_val,
        const long long* start, const long long* end,
        const long long* c_dims, const long long* c_strides,
        const bool* pasc, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long c_dim2 = *&c_dims[2];

    if(idx >= c_dim2) {
        return;
    }

    const bool asc = pasc[0];
    const long long start_interval = asc ? *&start[0] : *&end[0];
    const long long end_interval = asc ? *&end[0] : *&start[0];
    const double a = asc ? *&a_val[0] : *&b_val[0];
    const double b = asc ? *&b_val[0] : *&a_val[0];

    const long long stride_0 = *&c_strides[0];
    const long long stride_1 = *&c_strides[1];

    if(start_interval < 0 || end_interval < 0) {
        out[idx] = CUDART_NAN;
        return;
    }

    T vtot = 0;
    T vb;
    T va;
    for(int interval = start_interval; interval <= end_interval; interval++) {
        const double breakpoint = *&breakpoints[interval];
        if(interval == end_interval) {
            vb = eval_poly_1<T>(
                b - breakpoint, coef, interval, idx, -1, c_dims,
                stride_0, stride_1);
        } else {
            const double next_breakpoint = *&breakpoints[interval + 1];
            vb = eval_poly_1<T>(
                next_breakpoint - breakpoint, coef, interval,
                idx, -1, c_dims, stride_0, stride_1);
        }

        if(interval == start_interval) {
            va = eval_poly_1<T>(
                a - breakpoint, coef, interval, idx, -1, c_dims,
                stride_0, stride_1);
        } else {
            va = eval_poly_1<T>(
                0, coef, interval, idx, -1, c_dims,
                stride_0, stride_1);
        }

        vtot += (vb - va);
    }

    if(!asc) {
        vtot = -vtot;
    }

    out[idx] = vtot;

}
"""

PPOLY_MODULE = cupy.RawModule(
    code=PPOLY_KERNEL, options=('-std=c++11',),
    name_expressions=(
        [f'eval_ppoly<{type_name}>' for type_name in TYPES] +
        [f'fix_continuity<{type_name}>' for type_name in TYPES] +
        [f'integrate<{type_name}>' for type_name in TYPES]))

BPOLY_KERNEL = BASE_HEADERS + r"""
#include <cupy/complex.cuh>
#include <cupy/math_constants.h>

template<typename T>
__device__ T eval_bpoly1(
        const double s, const T* coef, const long long ci, const long long cj,
        const long long c_dims_0, const long long c_strides_0,
        const long long c_strides_1) {

    const long long k = c_dims_0 - 1;
    const double s1 = 1 - s;
    T res;

    const long long i0 = 0 * c_strides_0 + ci * c_strides_1 + cj;
    const long long i1 = 1 * c_strides_0 + ci * c_strides_1 + cj;
    const long long i2 = 2 * c_strides_0 + ci * c_strides_1 + cj;
    const long long i3 = 3 * c_strides_0 + ci * c_strides_1 + cj;

    if(k == 0) {
        res = coef[i0];
    } else if(k == 1) {
        res = coef[i0] * s1 + coef[i1] * s;
    } else if(k == 2) {
        res = coef[i0] * s1 * s1 + coef[i1] * 2.0 * s1 * s + coef[i2] * s * s;
    } else if(k == 3) {
        res = (coef[i0] * s1 * s1 * s1 + coef[i1] * 3.0 * s1 * s1 * s +
               coef[i2] * 3.0 * s1 * s * s + coef[i3] * s * s * s);
    } else {
        T comb = 1;
        res = 0;
        for(int j = 0; j < k + 1; j++) {
            const long long idx = j * c_strides_0 + ci * c_strides_1 + cj;
            res += (comb * pow(s, ((double) j)) * pow(s1, ((double) k) - j) *
                    coef[idx]);
            comb *= 1.0 * (k - j) / (j + 1.0);
        }
    }

    return res;
}

template<typename T>
__device__ T eval_bpoly1_deriv(
        const double s, const T* coef, const long long ci, const long long cj,
        int dx, T* wrk, const long long c_dims_0, const long long c_strides_0,
        const long long c_strides_1, const long long wrk_dims_0,
        const long long wrk_strides_0, const long long wrk_strides_1) {

    T res, term;
    double comb, poch;

    const long long k = c_dims_0 - 1;

    if(dx == 0) {
        res = eval_bpoly1<T>(s, coef, ci, cj, c_dims_0, c_strides_0,
                             c_strides_1);
    } else {
        poch = 1.0;
        for(int a = 0; a < dx; a++) {
            poch *= k - a;
        }

        term = 0;
        for(int a = 0; a < k - dx + 1; a++) {
            term = 0;
            comb = 1;
            for(int j = 0; j < dx + 1; j++) {
                const long long idx = (c_strides_0 * (j + a) +
                                       c_strides_1 * ci + cj);
                term += coef[idx] * pow(-1.0, ((double) (j + dx))) * comb;
                comb *= 1.0 * (dx - j) / (j + 1);
            }
            wrk[a] = term * poch;
        }

        res = eval_bpoly1<T>(s, wrk, 0, 0, wrk_dims_0, wrk_strides_0,
                             wrk_strides_1);
    }
    return res;
}

template<typename T>
__global__ void eval_bpoly(
        const T* coef, const double* breakpoints, const double* x,
        const long long* intervals, int dx, T* wrk, const long long* c_dims,
        const long long* c_strides, const long long* wrk_dims,
        const long long* wrk_strides, int num_x, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_x) {
        return;
    }

    double xp = *&x[idx];
    long long interval = *&intervals[idx];
    const int num_c = *&c_dims[2];

    const long long c_dims_0 = *&c_dims[0];
    const long long c_strides_0 = *&c_strides[0];
    const long long c_strides_1 = *&c_strides[1];

    const long long wrk_dims_0 = *&wrk_dims[0];
    const long long wrk_strides_0 = *&wrk_strides[0];
    const long long wrk_strides_1 = *&wrk_strides[1];

    if(interval < 0) {
        for(int j = 0; j < num_c; j++) {
            out[num_c * idx + j] = CUDART_NAN;
        }
        return;
    }

    const double ds = breakpoints[interval + 1] - breakpoints[interval];
    const double ds_dx = pow(ds, ((double) dx));
    T* off_wrk = wrk + idx * (c_dims_0 - dx);

    for(int j = 0; j < num_c; j++) {
        T res;
        const double s = (xp - breakpoints[interval]) / ds;
        if(dx == 0) {
            res = eval_bpoly1<T>(
                s, coef, interval, ((long long) (j)), c_dims_0, c_strides_0,
                c_strides_1);
        } else {
            res = eval_bpoly1_deriv<T>(
                s, coef, interval, ((long long) (j)), dx,
                off_wrk, c_dims_0, c_strides_0, c_strides_1,
                wrk_dims_0, wrk_strides_0, wrk_strides_1) / ds_dx;
        }
        out[num_c * idx + j] = res;
    }

}
"""

BPOLY_MODULE = cupy.RawModule(
    code=BPOLY_KERNEL, options=('-std=c++11',),
    name_expressions=(
        [f'eval_bpoly<{type_name}>' for type_name in TYPES]))


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


def _ppoly_evaluate(c, x, xp, dx, extrapolate, out):
    """
    Evaluate a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bool
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    # Determine if the breakpoints are in ascending order or descending one
    ascending = x[-1] >= x[0]

    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                    (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0],
                     ascending))

    # Compute coefficient displacement stride (in elements)
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize

    ppoly_kernel = _get_module_func(PPOLY_MODULE, 'eval_ppoly', c)
    ppoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                 (c, x, xp, intervals, dx, c_shape, c_strides,
                  xp.shape[0], out))


def _fix_continuity(c, x, order):
    """
    Make a piecewise polynomial continuously differentiable to given order.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.

        Coefficients c[-order-1:] are modified in-place.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    order : int
        Order up to which enforce piecewise differentiability.
    """
    # Compute coefficient displacement stride (in elements)
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize

    continuity_kernel = _get_module_func(PPOLY_MODULE, 'fix_continuity', c)
    continuity_kernel((1,), (1,),
                      (c, x, order, c_shape, c_strides, x.shape[0]))


def _integrate(c, x, a, b, extrapolate, out):
    """
    Compute integral over a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials
    a : double
        Start point of integration.
    b : double
        End point of integration.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (n,)
        Integral of the piecewise polynomial, assuming the polynomial
        is zero outside the range (x[0], x[-1]).
        This argument is modified in-place.
    """
    # Determine if the breakpoints are in ascending order or descending one
    ascending = x[-1] >= x[0]

    a = cupy.asarray([a], dtype=cupy.float64)
    b = cupy.asarray([b], dtype=cupy.float64)

    start_interval = cupy.empty(a.shape, dtype=cupy.int64)
    end_interval = cupy.empty(b.shape, dtype=cupy.int64)

    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((a.shape[0] + 128 - 1) // 128,), (128,),
                    (x, a, start_interval, extrapolate, a.shape[0], x.shape[0],
                     ascending))
    interval_kernel(((b.shape[0] + 128 - 1) // 128,), (128,),
                    (x, b, end_interval, extrapolate, b.shape[0], x.shape[0],
                     ascending))

    # Compute coefficient displacement stride (in elements)
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize

    int_kernel = _get_module_func(PPOLY_MODULE, 'integrate', c)
    int_kernel(((c.shape[2] + 128 - 1) // 128,), (128,),
               (c, x, a, b, start_interval, end_interval, c_shape, c_strides,
                ascending, out))


def _bpoly_evaluate(c, x, xp, dx, extrapolate, out):
    """
    Evaluate a Bernstein polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bool
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    # Determine if the breakpoints are in ascending order or descending one
    ascending = x[-1] >= x[0]

    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                    (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0],
                     ascending))

    # Compute coefficient displacement stride (in elements)
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize

    wrk = cupy.empty((xp.shape[0] * (c.shape[0] - dx), 1, 1),
                     dtype=_get_dtype(c))
    wrk_shape = cupy.asarray([c.shape[0] - dx, 1, 1], dtype=cupy.int64)
    wrk_strides = cupy.asarray(wrk.strides, dtype=cupy.int64) // wrk.itemsize

    bpoly_kernel = _get_module_func(BPOLY_MODULE, 'eval_bpoly', c)
    bpoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                 (c, x, xp, intervals, dx, wrk, c_shape, c_strides, wrk_shape,
                  wrk_strides, xp.shape[0], out))


class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = cupy.asarray(c)
        self.x = cupy.ascontiguousarray(x, dtype=cupy.float64)

        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("axis=%s must be between 0 and %s" %
                             (axis, self.c.ndim-1))

        self.axis = axis
        if axis != 0:
            # move the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = cupy.moveaxis(self.c, axis+1, 0)
            self.c = cupy.moveaxis(self.c, axis+1, 0)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = cupy.diff(self.x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if (cupy.issubdtype(dtype, cupy.complexfloating)
                or cupy.issubdtype(self.c.dtype, cupy.complexfloating)):
            return cupy.complex_
        else:
            return cupy.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.
        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        """

        c = cupy.asarray(c)
        x = cupy.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("Shapes of x {} and c {} are incompatible"
                             .format(x.shape, c.shape))
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("Shapes of c {} and self.c {} are incompatible"
                             .format(c.shape, self.c.shape))

        if c.size == 0:
            return

        dx = cupy.diff(x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = cupy.zeros(
            (k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
            dtype=dtype)

        if action == 'append':
            c2[k2 - self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2 - c.shape[0]:, self.c.shape[1]:] = c
            self.x = cupy.r_[self.x, x]
        elif action == 'prepend':
            c2[k2 - self.c.shape[0]:, :c.shape[1]] = c
            c2[k2 - c.shape[0]:, c.shape[1]:] = self.c
            self.x = cupy.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = cupy.ascontiguousarray(x.ravel(), dtype=cupy.float_)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        out = cupy.empty((len(x), int(np.prod(self.c.shape[2:]))),
                         dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            dims = list(range(out.ndim))
            dims = (dims[x_ndim:x_ndim + self.axis] + dims[:x_ndim] +
                    dims[x_ndim + self.axis:])
            out = out.transpose(dims)
        return out


class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints
    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i]) ** (k - m) for m in range(k + 1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.

    .. seealso:: :class:`scipy.interpolate.BSpline`
    """

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the
            derivative of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(cupy.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,)*(c2.ndim-1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.
        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        c = cupy.zeros(
            (self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
            dtype=self.c.dtype)
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = spec.poch(cupy.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim-1)]

        # fix continuity of added degrees of freedom
        self._ensure_c_contiguous()
        _fix_continuity(c.reshape(c.shape[0], c.shape[1], -1),
                        self.x, nu - 1)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        range_int = cupy.empty(
            (int(np.prod(self.c.shape[2:])),), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        # Compute the integral.
        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                _integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)

            # Map a to [xs, xe], b is always a + left.
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            remainder_int = cupy.empty_like(range_int)
            if b <= xe:
                _integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int

                _integrate(
                    self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                    self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            _integrate(
                self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                self.x, a, b, bool(extrapolate), out=range_int)

        # Return
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0., discontinuity=True, extrapolate=None):
        """
        Find real solutions of the equation ``pp(x) == y``.

        Parameters
        ----------
        y : float, optional
            Right-hand side. Default is zero.
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.
        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.
        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        At the moment, there is not an actual implementation.
        """
        raise NotImplementedError(
            'At the moment there is not a GPU implementation for solve')

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to return roots from the polynomial
            extrapolated based on first and last intervals, 'periodic' works
            the same as False. If None (default), use `self.extrapolate`.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).
            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        See Also
        --------
        PPoly.solve
        """
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as a (knots, coefficients, degree) tuple or
            a BSpline object.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck

        spl = BSpline(t, c, k, extrapolate=extrapolate)
        cvals = cupy.empty((k + 1, len(t) - 1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = spl(t[:-1], nu=m)
            cvals[k - m, :] = y / spec.gamma(m + 1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(bp, BPoly):
            raise TypeError(".from_bernstein_basis only accepts BPoly "
                            "instances. Got %s instead." % type(bp))

        dx = cupy.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,)*(bp.c.ndim-2)

        c = cupy.zeros_like(bp.c)
        for a in range(k+1):
            factor = (-1)**a * _comb(k, a) * bp.c[a]
            for s in range(a, k+1):
                val = _comb(k-a, s-a) * (-1)**s
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)


class BPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the

    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial
    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf
    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546,
           `10.1155/2011/829543 <https://doi.org/10.1155/2011/829543>`_.

    Examples
    --------
    >>> from cupyx.scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) +
               3 \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2
    """

    def _evaluate(self, x, nu, extrapolate, out):
        # check derivative order
        if nu < 0:
            raise NotImplementedError(
                "Cannot do antiderivatives in the B-basis yet.")

        _bpoly_evaluate(
            self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
            self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k - nu representing the derivative of
            this polynomial.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            # For a polynomial
            #    B(x) = \sum_{a=0}^{k} c_a b_{a, k}(x),
            # we use the fact that
            #   b'_{a, k} = k ( b_{a-1, k-1} - b_{a, k-1} ),
            # which leads to
            #   B'(x) = \sum_{a=0}^{k-1} (c_{a+1} - c_a) b_{a, k-1}
            #
            # finally, for an interval [y, y + dy] with dy != 1,
            # we need to correct for an extra power of dy

            rest = (None,) * (self.c.ndim-2)

            k = self.c.shape[0] - 1
            dx = cupy.diff(self.x)[(None, slice(None))+rest]
            c2 = k * cupy.diff(self.c, axis=0) / dx

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        bp : BPoly
            Piecewise polynomial of order k + nu representing the
            antiderivative of this polynomial.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp

        # Construct the indefinite integrals on individual intervals
        c, x = self.c, self.x
        k = c.shape[0]
        c2 = cupy.zeros((k+1,) + c.shape[1:], dtype=c.dtype)

        c2[1:, ...] = cupy.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,)*(c.ndim-2)]

        # Now fix continuity: on the very first interval, take the integration
        # constant to be zero; on an interval [x_j, x_{j+1}) with j>0,
        # the integration constant is then equal to the jump of the `bp`
        # at x_j.
        # The latter is given by the coefficient of B_{n+1, n+1}
        # *on the previous interval* (other B. polynomials are zero at the
        # breakpoint). Finally, use the fact that BPs form a partition of
        # unity.
        c2[:, 1:] += cupy.cumsum(c2[k, :], axis=0)[:-1]

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, x, extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : {bool, 'periodic', None}, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs. If 'periodic', periodic
            extrapolation is used. If None (default), use `self.extrapolate`.

        Returns
        -------
        array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        # XXX: can probably use instead the fact that
        # \int_0^{1} B_{j, n}(x) \dx = 1/(n+1)
        ib = self.antiderivative()
        if extrapolate is None:
            extrapolate = self.extrapolate

        # ib.extrapolate shouldn't be 'periodic', it is converted to
        # False for 'periodic. in antiderivative() call.
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            # For simplicity and clarity convert to a <= b case.
            if a <= b:
                sign = 1
            else:
                a, b = b, a
                sign = -1

            xs, xe = self.x[0], self.x[-1]
            period = xe - xs
            interval = b - a
            n_periods, left = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))

            # Map a and b to [xs, xe].
            a = xs + (a - xs) % period
            b = a + left

            # If b <= xe then we need to integrate over [a, b], otherwise
            # over [a, xe] and from xs to what is remained.
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)

            return sign * res
        else:
            return ib(b) - ib(a)

    def extend(self, c, x):
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @staticmethod
    def _raise_degree(c, d):
        r"""
        Raise a degree of a polynomial in the Bernstein basis.

        Given the coefficients of a polynomial degree `k`, return (the
        coefficients of) the equivalent polynomial of degree `k+d`.

        Parameters
        ----------
        c : array_like
            coefficient array, 1-D
        d : integer

        Returns
        -------
        array
            coefficient array, 1-D array of length `c.shape[0] + d`

        Notes
        -----
        This uses the fact that a Bernstein polynomial `b_{a, k}` can be
        identically represented as a linear combination of polynomials of
        a higher degree `k+d`:

            .. math:: b_{a, k} = comb(k, a) \sum_{j=0}^{d} b_{a+j, k+d} \
                                 comb(d, j) / comb(k+d, a+j)
        """
        if d == 0:
            return c

        k = c.shape[0] - 1
        out = cupy.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)

        for a in range(c.shape[0]):
            f = c[a] * _comb(k, a)
            for j in range(d + 1):
                out[a + j] += f * _comb(d, j) / _comb(k + d, a + j)
        return out

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        """
        Construct a piecewise polynomial in Bernstein basis
        from a power basis polynomial.

        Parameters
        ----------
        pp : PPoly
            A piecewise polynomial in the power basis
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.
        """
        if not isinstance(pp, PPoly):
            raise TypeError(".from_power_basis only accepts PPoly instances. "
                            "Got %s instead." % type(pp))

        dx = cupy.diff(pp.x)
        k = pp.c.shape[0] - 1   # polynomial order

        rest = (None,)*(pp.c.ndim-2)

        c = cupy.zeros_like(pp.c)
        for a in range(k+1):
            factor = pp.c[a] / _comb(k, k-a) * dx[(slice(None),)+rest]**(k-a)
            for j in range(k-a, k+1):
                c[j] += factor * _comb(j, k-a)

        if extrapolate is None:
            extrapolate = pp.extrapolate

        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        """
        Construct a piecewise polynomial in the Bernstein basis,
        compatible with the specified values and derivatives at breakpoints.

        Parameters
        ----------
        xi : array_like
            sorted 1-D array of x-coordinates
        yi : array_like or list of array_likes
            ``yi[i][j]`` is the ``j`` th derivative known at ``xi[i]``
        orders : None or int or array_like of ints. Default: None.
            Specifies the degree of local polynomials. If not None, some
            derivatives are ignored.
        extrapolate : bool or 'periodic', optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used. Default is True.

        Notes
        -----
        If ``k`` derivatives are specified at a breakpoint ``x``, the
        constructed polynomial is exactly ``k`` times continuously
        differentiable at ``x``, unless the ``order`` is provided explicitly.
        In the latter case, the smoothness of the polynomial at
        the breakpoint is controlled by the ``order``.

        Deduces the number of derivatives to match at each end
        from ``order`` and the number of derivatives available. If
        possible it uses the same number of derivatives from
        each end; if the number is odd it tries to take the
        extra one from y2. In any case if not enough derivatives
        are available at one end or another it draws enough to
        make up the total from the other end.

        If the order is too high and not enough derivatives are available,
        an exception is raised.

        Examples
        --------
        >>> from cupyx.scipy.interpolate import BPoly
        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])

        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`
        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`

        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])

        Creates a piecewise polynomial `f(x)`, such that
        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.
        Based on the number of derivatives provided, the order of the
        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.
        Notice that no restriction is imposed on the derivatives at
        ``x = 1`` and ``x = 2``.

        Indeed, the explicit form of the polynomial is::

            f(x) = | x * (1 - x),  0 <= x < 1
                   | 2 * (x - 1),  1 <= x <= 2

        So that f'(1-0) = -1 and f'(1+0) = 2
        """
        xi = cupy.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError("xi and yi need to have the same length")
        if cupy.any(xi[1:] - xi[:1] <= 0):
            raise ValueError("x coordinates are not in increasing order")

        # number of intervals
        m = len(xi) - 1

        # global poly order is k-1, local orders are <=k and can vary
        try:
            k = max(len(yi[i]) + len(yi[i+1]) for i in range(m))
        except TypeError as e:
            raise ValueError(
                "Using a 1-D array for y? Please .reshape(-1, 1)."
            ) from e

        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (int, cupy.integer)):
                orders = [orders] * m
            k = max(k, max(orders))

            if any(o <= 0 for o in orders):
                raise ValueError("Orders must be positive.")

        c = []
        for i in range(m):
            y1, y2 = yi[i], yi[i+1]
            if orders[i] is None:
                n1, n2 = len(y1), len(y2)
            else:
                n = orders[i]+1
                n1 = min(n//2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1+n2 != n:
                    mesg = ("Point %g has %d derivatives, point %g"
                            " has %d derivatives, but order %d requested" % (
                                xi[i], len(y1), xi[i+1], len(y2), orders[i]))
                    raise ValueError(mesg)

                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError("`order` input incompatible with"
                                     " length y1 or y2.")

            b = BPoly._construct_from_derivatives(xi[i], xi[i+1],
                                                  y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)

        c = cupy.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        r"""
        Compute the coefficients of a polynomial in the Bernstein basis
        given the values and derivatives at the edges.

        Return the coefficients of a polynomial in the Bernstein basis
        defined on ``[xa, xb]`` and having the values and derivatives at the
        endpoints `xa` and `xb` as specified by `ya`` and `yb`.

        The polynomial constructed is of the minimal possible degree, i.e.,
        if the lengths of `ya` and `yb` are `na` and `nb`, the degree
        of the polynomial is ``na + nb - 1``.

        Parameters
        ----------
        xa : float
            Left-hand end point of the interval
        xb : float
            Right-hand end point of the interval
        ya : array_like
            Derivatives at `xa`. `ya[0]` is the value of the function, and
            `ya[i]` for ``i > 0`` is the value of the ``i``th derivative.
        yb : array_like
            Derivatives at `xb`.

        Returns
        -------
        array
            coefficient array of a polynomial having specified derivatives

        Notes
        -----
        This uses several facts from life of Bernstein basis functions.
        First of all,

            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})

        If B(x) is a linear combination of the form

            .. math:: B(x) = \sum_{a=0}^{n} c_a b_{a, n},

        then :math: B'(x) = n \sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.
        Iterating the latter one, one finds for the q-th derivative

            .. math:: B^{q}(x) = n!/(n-q)! \sum_{a=0}^{n-q} Q_a b_{a, n-q},

        with

            .. math:: Q_a = \sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}

        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and
        `c_q` are found one by one by iterating `q = 0, ..., na`.

        At ``x = xb`` it's the same with ``a = n - q``.
        """
        ya, yb = cupy.asarray(ya), cupy.asarray(yb)
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError('Shapes of ya {} and yb {} are incompatible'
                             .format(ya.shape, yb.shape))

        dta, dtb = ya.dtype, yb.dtype
        if (cupy.issubdtype(dta, cupy.complexfloating) or
                cupy.issubdtype(dtb, cupy.complexfloating)):
            dt = cupy.complex_
        else:
            dt = cupy.float_

        na, nb = len(ya), len(yb)
        n = na + nb

        c = cupy.empty((na+nb,) + ya.shape[1:], dtype=dt)

        # compute coefficients of a polynomial degree na+nb-1
        # walk left-to-right
        for q in range(0, na):
            c[q] = ya[q] / spec.poch(n - q, q) * (xb - xa)**q
            for j in range(0, q):
                c[q] -= (-1)**(j+q) * _comb(q, j) * c[j]

        # now walk right-to-left
        for q in range(0, nb):
            c[-q-1] = yb[q] / spec.poch(n - q, q) * (-1)**q * (xb - xa)**q
            for j in range(0, q):
                c[-q-1] -= (-1)**(j+1) * _comb(q, j+1) * c[-q+j]

        return c
