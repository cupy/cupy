"""
Peak finding functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import math
import cupy

from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

from cupyx import jit


def _get_typename(dtype):
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            # 'half' in name_expressions weirdly raises
            # HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID in getLoweredName() on
            # ROCm
            typename = '__half'
        else:
            typename = 'half'
    return typename


FLOAT_TYPES = [cupy.float16, cupy.float32, cupy.float64]
INT_TYPES = [cupy.int8, cupy.int16, cupy.int32, cupy.int64]
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
FLOAT_INT_TYPES = FLOAT_TYPES + INT_TYPES  # type: ignore
TYPES = FLOAT_INT_TYPES + UNSIGNED_TYPES  # type: ignore
TYPE_NAMES = [_get_typename(t) for t in TYPES]
FLOAT_INT_NAMES = [_get_typename(t) for t in FLOAT_INT_TYPES]

_modedict = {
    cupy.less: 0,
    cupy.greater: 1,
    cupy.less_equal: 2,
    cupy.greater_equal: 3,
    cupy.equal: 4,
    cupy.not_equal: 5,
}

if runtime.is_hip:
    PEAKS_KERNEL_BASE = r"""
    #include <hip/hip_runtime.h>
"""
else:
    PEAKS_KERNEL_BASE = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

PEAKS_KERNEL = PEAKS_KERNEL_BASE + r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T>
__global__ void local_maxima_1d(
        const int n, const T* __restrict__ x, long long* midpoints,
        long long* left_edges, long long* right_edges) {

    const int orig_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx = orig_idx + 1;

    if(idx >= n - 1) {
        return;
    }

    long long midpoint = -1;
    long long left = -1;
    long long right = -1;

    if(x[idx - 1] < x[idx]) {
        int i_ahead = idx + 1;

        while(i_ahead < n - 1 && x[i_ahead] == x[idx]) {
            i_ahead++;
        }

        if(x[i_ahead] < x[idx]) {
            left = idx;
            right = i_ahead - 1;
            midpoint = (left + right) / 2;
        }
    }

    midpoints[orig_idx] = midpoint;
    left_edges[orig_idx] = left;
    right_edges[orig_idx] = right;
}

template<typename T>
__global__ void peak_prominences(
        const int n, const int n_peaks, const T* __restrict__ x,
        const long long* __restrict__ peaks, const long long wlen,
        T* prominences, long long* left_bases, long long* right_bases) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n_peaks) {
        return;
    }

    const long long peak = peaks[idx];
    long long i_min = 0;
    long long i_max = n - 1;

    if(wlen >= 2) {
        i_min = max(peak - wlen / 2, i_min);
        i_max = min(peak + wlen / 2, i_max);
    }

    left_bases[idx] = peak;
    long long i = peak;
    T left_min = x[peak];

    while(i_min <= i && x[i] <= x[peak]) {
        if(x[i] < left_min) {
            left_min = x[i];
            left_bases[idx] = i;
        }
        i--;
    }

    right_bases[idx] = peak;
    i = peak;
    T right_min = x[peak];

    while(i <= i_max && x[i] <= x[peak]) {
        if(x[i] < right_min) {
            right_min = x[i];
            right_bases[idx] = i;
        }
        i++;
    }

    prominences[idx] = x[peak] - max(left_min, right_min);
}

template<>
__global__ void peak_prominences<half>(
        const int n, const int n_peaks, const half* __restrict__ x,
        const long long* __restrict__ peaks, const long long wlen,
        half* prominences, long long* left_bases, long long* right_bases) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n_peaks) {
        return;
    }

    const long long peak = peaks[idx];
    long long i_min = 0;
    long long i_max = n - 1;

    if(wlen >= 2) {
        i_min = max(peak - wlen / 2, i_min);
        i_max = min(peak + wlen / 2, i_max);
    }

    left_bases[idx] = peak;
    long long i = peak;
    half left_min = x[peak];

    while(i_min <= i && x[i] <= x[peak]) {
        if(x[i] < left_min) {
            left_min = x[i];
            left_bases[idx] = i;
        }
        i--;
    }

    right_bases[idx] = peak;
    i = peak;
    half right_min = x[peak];

    while(i <= i_max && x[i] <= x[peak]) {
        if(x[i] < right_min) {
            right_min = x[i];
            right_bases[idx] = i;
        }
        i++;
    }

    prominences[idx] = x[peak] - __hmax(left_min, right_min);
}

template<typename T>
__global__ void peak_widths(
        const int n, const T* __restrict__ x,
        const long long* __restrict__ peaks,
        const double rel_height,
        const T* __restrict__ prominences,
        const long long* __restrict__ left_bases,
        const long long* __restrict__ right_bases,
        double* widths, double* width_heights,
        double* left_ips, double* right_ips) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n) {
        return;
    }

    long long i_min = left_bases[idx];
    long long i_max = right_bases[idx];
    long long peak = peaks[idx];

    double height = x[peak] - prominences[idx] * rel_height;
    width_heights[idx] = height;

    // Find intersection point on left side
    long long i = peak;
    while (i_min < i && height < x[i]) {
        i--;
    }

    double left_ip = (double) i;
    if(x[i] < height) {
        // Interpolate if true intersection height is between samples
        left_ip += (height - x[i]) / (x[i + 1] - x[i]);
    }

    // Find intersection point on right side
    i = peak;
    while(i < i_max && height < x[i]) {
        i++;
    }

    double right_ip = (double) i;
    if(x[i] < height) {
        // Interpolate if true intersection height is between samples
        right_ip -= (height - x[i]) / (x[i - 1] - x[i]);
    }

    widths[idx] = right_ip - left_ip;
    left_ips[idx] = left_ip;
    right_ips[idx] = right_ip;
}

template<>
__global__ void peak_widths<half>(
        const int n, const half* __restrict__ x,
        const long long* __restrict__ peaks,
        const double rel_height,
        const half* __restrict__ prominences,
        const long long* __restrict__ left_bases,
        const long long* __restrict__ right_bases,
        double* widths, double* width_heights,
        double* left_ips, double* right_ips) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n) {
        return;
    }

    long long i_min = left_bases[idx];
    long long i_max = right_bases[idx];
    long long peak = peaks[idx];

    double height = ((double) x[peak]) - ((double) prominences[idx]) * rel_height;
    width_heights[idx] = height;

    // Find intersection point on left side
    long long i = peak;
    while (i_min < i && height < ((double) x[i])) {
        i--;
    }

    double left_ip = (double) i;
    if(((double) x[i]) < height) {
        // Interpolate if true intersection height is between samples
        left_ip += (height - ((double) x[i])) / ((double) (x[i + 1] - x[i]));
    }

    // Find intersection point on right side
    i = peak;
    while(i < i_max && height < ((double) x[i])) {
        i++;
    }

    double right_ip = (double) i;
    if(((double) x[i]) < height) {
        // Interpolate if true intersection height is between samples
        right_ip -= (height - ((double) x[i])) / ((double) (x[i - 1] - x[i]));
    }

    widths[idx] = right_ip - left_ip;
    left_ips[idx] = left_ip;
    right_ips[idx] = right_ip;
}
"""  # NOQA

PEAKS_MODULE = cupy.RawModule(
    code=PEAKS_KERNEL, options=('-std=c++11',),
    name_expressions=[f'local_maxima_1d<{x}>' for x in TYPE_NAMES] +
    [f'peak_prominences<{x}>' for x in TYPE_NAMES] +
    [f'peak_widths<{x}>' for x in TYPE_NAMES])


ARGREL_KERNEL = r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T>
__device__ __forceinline__ bool less( const T &a, const T &b ) {
    return ( a < b );
}

template<typename T>
__device__ __forceinline__ bool greater( const T &a, const T &b ) {
    return ( a > b );
}

template<typename T>
__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {
    return ( a <= b );
}

template<typename T>
__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {
    return ( a >= b );
}

template<typename T>
__device__ __forceinline__ bool equal( const T &a, const T &b ) {
    return ( a == b );
}

template<typename T>
__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {
    return ( a != b );
}

__device__ __forceinline__ void clip_plus(
        const bool &clip, const int &n, int &plus ) {
    if ( clip ) {
        if ( plus >= n ) {
            plus = n - 1;
        }
    } else {
        if ( plus >= n ) {
            plus -= n;
        }
    }
}

__device__ __forceinline__ void clip_minus(
        const bool &clip, const int &n, int &minus ) {
    if ( clip ) {
        if ( minus < 0 ) {
            minus = 0;
        }
    } else {
        if ( minus < 0 ) {
            minus += n;
        }
    }
}

template<typename T>
__device__ bool compare(const int comp, const T &a, const T &b) {
    if(comp == 0) {
        return less(a, b);
    } else if(comp == 1) {
        return greater(a, b);
    } else if(comp == 2) {
        return less_equal(a, b);
    } else if(comp == 3) {
        return greater_equal(a, b);
    } else if(comp == 4) {
        return equal(a, b);
    } else {
        return not_equal(a, b);
    }
}

template<typename T>
__global__ void boolrelextrema_1D( const int  n,
                                   const int  order,
                                   const bool clip,
                                   const int  comp,
                                   const T *__restrict__ inp,
                                   bool *__restrict__ results) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < n; tid += stride ) {

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus { tid + o };
            int minus { tid - o };

            clip_plus( clip, n, plus );
            clip_minus( clip, n, minus );

            temp &= compare<T>( comp,  data, inp[plus] );
            temp &= compare<T>( comp, data, inp[minus] );
        }
        results[tid] = temp;
    }
}

template<typename T>
__global__ void boolrelextrema_2D( const int  in_x,
                                   const int  in_y,
                                   const int  order,
                                   const bool clip,
                                   const int  comp,
                                   const int  axis,
                                   const T *__restrict__ inp,
                                   bool *__restrict__ results) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    if ( ( tx < in_y ) && ( ty < in_x ) ) {
        int tid { tx * in_x + ty };

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {

            int plus {};
            int minus {};

            if ( axis == 0 ) {
                plus  = tx + o;
                minus = tx - o;

                clip_plus( clip, in_y, plus );
                clip_minus( clip, in_y, minus );

                plus  = plus * in_x + ty;
                minus = minus * in_x + ty;
            } else {
                plus  = ty + o;
                minus = ty - o;

                clip_plus( clip, in_x, plus );
                clip_minus( clip, in_x, minus );

                plus  = tx * in_x + plus;
                minus = tx * in_x + minus;
            }

            temp &= compare<T>( comp, data, inp[plus] );
            temp &= compare<T>( comp, data, inp[minus] );
        }
        results[tid] = temp;
    }
}
"""


ARGREL_MODULE = cupy.RawModule(
    code=ARGREL_KERNEL, options=('-std=c++11',),
    name_expressions=[f'boolrelextrema_1D<{x}>' for x in FLOAT_INT_NAMES] +
    [f'boolrelextrema_2D<{x}>' for x in FLOAT_INT_NAMES])


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def _local_maxima_1d(x):
    samples = x.shape[0] - 2
    block_sz = 128
    n_blocks = (samples + block_sz - 1) // block_sz

    midpoints = cupy.empty(samples, dtype=cupy.int64)
    left_edges = cupy.empty(samples, dtype=cupy.int64)
    right_edges = cupy.empty(samples, dtype=cupy.int64)

    local_max_kernel = _get_module_func(PEAKS_MODULE, 'local_maxima_1d', x)
    local_max_kernel((n_blocks,), (block_sz,),
                     (x.shape[0], x, midpoints, left_edges, right_edges))

    pos_idx = midpoints > 0
    midpoints = midpoints[pos_idx]
    left_edges = left_edges[pos_idx]
    right_edges = right_edges[pos_idx]

    return midpoints, left_edges, right_edges


def _unpack_condition_args(interval, x, peaks):
    """
    Parse condition arguments for `find_peaks`.

    Parameters
    ----------
    interval : number or ndarray or sequence
        Either a number or ndarray or a 2-element sequence of the former. The
        first value is always interpreted as `imin` and the second,
        if supplied, as `imax`.
    x : ndarray
        The signal with `peaks`.
    peaks : ndarray
        An array with indices used to reduce `imin` and / or `imax` if those
        are arrays.

    Returns
    -------
    imin, imax : number or ndarray or None
        Minimal and maximal value in `argument`.

    Raises
    ------
    ValueError :
        If interval border is given as array and its size does not match the
        size of `x`.
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    # Reduce arrays if arrays
    if isinstance(imin, cupy.ndarray):
        if imin.size != x.size:
            raise ValueError(
                'array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, cupy.ndarray):
        if imax.size != x.size:
            raise ValueError(
                'array size of upper interval border must match x')
        imax = imax[peaks]

    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):
    """
    Evaluate where the generic property of peaks confirms to an interval.

    Parameters
    ----------
    peak_properties : ndarray
        An array with properties for each peak.
    pmin : None or number or ndarray
        Lower interval boundary for `peak_properties`. ``None``
        is interpreted as an open border.
    pmax : None or number or ndarray
        Upper interval boundary for `peak_properties`. ``None``
        is interpreted as an open border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peak_properties` confirms
        to the interval.

    See Also
    --------
    find_peaks

    """
    keep = cupy.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= (pmin <= peak_properties)
    if pmax is not None:
        keep &= (peak_properties <= pmax)
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):
    """
    Evaluate which peaks fulfill the threshold condition.

    Parameters
    ----------
    x : ndarray
        A 1-D array which is indexable by `peaks`.
    peaks : ndarray
        Indices of peaks in `x`.
    tmin, tmax : scalar or ndarray or None
         Minimal and / or maximal required thresholds. If supplied as ndarrays
         their size must match `peaks`. ``None`` is interpreted as an open
         border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peaks` fulfill the threshold
        condition.
    left_thresholds, right_thresholds : ndarray
        Array matching `peak` containing the thresholds of each peak on
        both sides.

    """
    # Stack thresholds on both sides to make min / max operations easier:
    # tmin is compared with the smaller, and tmax with the greater thresold to
    # each peak's side
    stacked_thresholds = cupy.vstack([x[peaks] - x[peaks - 1],
                                      x[peaks] - x[peaks + 1]])
    keep = cupy.ones(peaks.size, dtype=bool)
    if tmin is not None:
        min_thresholds = cupy.min(stacked_thresholds, axis=0)
        keep &= (tmin <= min_thresholds)
    if tmax is not None:
        max_thresholds = cupy.max(stacked_thresholds, axis=0)
        keep &= (max_thresholds <= tmax)

    return keep, stacked_thresholds[0], stacked_thresholds[1]


def _select_by_peak_distance(peaks, priority, distance):
    """
    Evaluate which peaks fulfill the distance condition.

    Parameters
    ----------
    peaks : ndarray
        Indices of peaks in `vector`.
    priority : ndarray
        An array matching `peaks` used to determine priority of each peak. A
        peak with a higher priority value is kept over one with a lower one.
    distance : np.float64
        Minimal distance that peaks must be spaced.

    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.

    Notes
    -----
    Declaring the input arrays as C-contiguous doesn't seem to have performance
    advantages.
    """
    peaks_size = peaks.shape[0]
    # Round up because actual peak distance can only be natural number
    distance_ = cupy.ceil(distance)
    keep = cupy.ones(peaks_size, dtype=cupy.bool_)  # Prepare array of flags

    # Create map from `i` (index for `peaks` sorted by `priority`) to `j`
    # (index for `peaks` sorted by position). This allows to iterate `peaks`
    # and `keep` with `j` by order of `priority` while still maintaining the
    # ability to step to neighbouring peaks with (`j` + 1) or (`j` - 1).
    priority_to_position = cupy.argsort(priority)

    # Highest priority first -> iterate in reverse order (decreasing)

    # NOTE: There's not an alternative way to do this procedure in a parallel
    # fashion, since discarding a peak requires to know if there's a valid
    # neighbour that subsumes it, which in turn requires to know
    # if that neighbour is valid. If it was to done in parallel, there would be
    # tons of repeated computations per peak, thus increasing the total runtime
    # per peak compared to a sequential implementation.
    for i in range(peaks_size - 1, -1, -1):
        # "Translate" `i` to `j` which points to current peak whose
        # neighbours are to be evaluated
        j = priority_to_position[i]
        if keep[j] == 0:
            # Skip evaluation for peak already marked as "don't keep"
            continue

        k = j - 1
        # Flag "earlier" peaks for removal until minimal distance is exceeded
        while 0 <= k and peaks[j] - peaks[k] < distance_:
            keep[k] = 0
            k -= 1

        k = j + 1
        # Flag "later" peaks for removal until minimal distance is exceeded
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = 0
            k += 1
    return keep


def _arg_x_as_expected(value):
    """Ensure argument `x` is a 1-D C-contiguous array.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array.
    """
    value = cupy.asarray(value, order='C')
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value


def _arg_wlen_as_expected(value):
    """Ensure argument `wlen` is of type `np.intp` and larger than 1.

    Used in `peak_prominences` and `peak_widths`.

    Returns
    -------
    value : np.intp
        The original `value` rounded up to an integer or -1 if `value` was
        None.
    """
    if value is None:
        # _peak_prominences expects an intp; -1 signals that no value was
        # supplied by the user
        value = -1
    elif 1 < value:
        # Round up to a positive integer
        if not cupy.can_cast(value, cupy.int64, "safe"):
            value = math.ceil(value)
        value = int(value)
    else:
        raise ValueError('`wlen` must be larger than 1, was {}'
                         .format(value))
    return value


def _arg_peaks_as_expected(value):
    """Ensure argument `peaks` is a 1-D C-contiguous array of dtype('int64').

    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible
    with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('int64').
    """
    value = cupy.asarray(value)
    if value.size == 0:
        # Empty arrays default to cupy.float64 but are valid input
        value = cupy.array([], dtype=cupy.int64)
    try:
        # Safely convert to C-contiguous array of type cupy.int64
        value = value.astype(cupy.int64, order='C', copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value


@jit.rawkernel()
def _check_prominence_invalid(n, peaks, left_bases, right_bases, out):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    i_min = left_bases[tid]
    i_max = right_bases[tid]
    peak = peaks[tid]
    valid = 0 <= i_min and i_min <= peak and peak <= i_max and i_max < n
    out[tid] = not valid


def _peak_prominences(x, peaks, wlen=None, check=False):
    if check and cupy.any(cupy.logical_or(peaks < 0, peaks > x.shape[0] - 1)):
        raise ValueError('peaks are not a valid index')

    prominences = cupy.empty(peaks.shape[0], dtype=x.dtype)
    left_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)
    right_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)

    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz

    peak_prom_kernel = _get_module_func(PEAKS_MODULE, 'peak_prominences', x)
    peak_prom_kernel(
        (n_blocks,), (block_sz,),
        (x.shape[0], n, x, peaks, wlen, prominences, left_bases, right_bases))

    return prominences, left_bases, right_bases


def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases,
                 check=False):
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    if prominences is None:
        raise TypeError('prominences must not be None')
    if left_bases is None:
        raise TypeError('left_bases must not be None')
    if right_bases is None:
        raise TypeError('right_bases must not be None')
    if not (peaks.shape[0] == prominences.shape[0] == left_bases.shape[0]
            == right_bases.shape[0]):
        raise ValueError("arrays in `prominence_data` must have the same "
                         "shape as `peaks`")

    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz

    if check and n > 0:
        invalid = cupy.zeros(n, dtype=cupy.bool_)
        _check_prominence_invalid(
            (n_blocks,), (block_sz,),
            (x.shape[0], peaks, left_bases, right_bases, invalid))
        if cupy.any(invalid):
            raise ValueError("prominence data is invalid")

    widths = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    width_heights = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    left_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    right_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)

    peak_widths_kernel = _get_module_func(PEAKS_MODULE, 'peak_widths', x)
    peak_widths_kernel(
        (n_blocks,), (block_sz,),
        (n, x, peaks, rel_height, prominences, left_bases, right_bases,
         widths, width_heights, left_ips, right_ips))
    return widths, width_heights, left_ips, right_ips


def peak_prominences(x, peaks, wlen=None):
    """
    Calculate the prominence of each peak in a signal.

    The prominence of a peak measures how much a peak stands out from the
    surrounding baseline of the signal and is defined as the vertical distance
    between the peak and its lowest contour line.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    wlen : int, optional
        A window length in samples that optionally limits the evaluated area
        for each peak to a subset of `x`. The peak is always placed in the
        middle of the window therefore the given length is rounded up to the
        next odd integer. This parameter can speed up the calculation
        (see Notes).

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.
        The higher base of each pair is a peak's lowest contour line.

    Raises
    ------
    ValueError
        If a value in `peaks` is an invalid index for `x`.

    Warns
    -----
    PeakPropertyWarning
        For indices in `peaks` that don't point to valid local maxima in `x`,
        the returned prominence will be 0 and this warning is raised. This
        also happens if `wlen` is smaller than the plateau size of a peak.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_widths
        Calculate the width of peaks.

    Notes
    -----
    Strategy to compute a peak's prominence:

    1. Extend a horizontal line from the current peak to the left and right
       until the line either reaches the window border (see `wlen`) or
       intersects the signal again at the slope of a higher peak. An
       intersection with a peak of the same height is ignored.
    2. On each side find the minimal signal value within the interval defined
       above. These points are the peak's bases.
    3. The higher one of the two bases marks the peak's lowest contour line.
       The prominence can then be calculated as the vertical difference between
       the peaks height itself and its lowest contour line.

    Searching for the peak's bases can be slow for large `x` with periodic
    behavior because large chunks or even the full signal need to be evaluated
    for the first algorithmic step. This evaluation area can be limited with
    the parameter `wlen` which restricts the algorithm to a window around the
    current peak and can shorten the calculation time if the window length is
    short in relation to `x`.
    However, this may stop the algorithm from finding the true global contour
    line if the peak's true bases are outside this window. Instead, a higher
    contour line is found within the restricted window leading to a smaller
    calculated prominence. In practice, this is only relevant for the highest
    set of peaks in `x`. This behavior may even be used intentionally to
    calculate "local" prominences.

    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    wlen = _arg_wlen_as_expected(wlen)
    return _peak_prominences(x, peaks, wlen, check=True)


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """
    Calculate the width of each peak in a signal.

    This function calculates the width of a peak in samples at a relative
    distance to the peak's height and prominence.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    rel_height : float, optional
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence. 1.0 calculates the width of the peak at
        its lowest contour line while 0.5 evaluates at half the prominence
        height. Must be at least 0. See notes for further explanation.
    prominence_data : tuple, optional
        A tuple of three arrays matching the output of `peak_prominences` when
        called with the same arguments `x` and `peaks`. This data are
        calculated internally if not provided.
    wlen : int, optional
        A window length in samples passed to `peak_prominences` as an optional
        argument for internal calculation of `prominence_data`. This argument
        is ignored if `prominence_data` is given.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.

    Raises
    ------
    ValueError
        If `prominence_data` is supplied but doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak,
        has the wrong dtype, is not C-contiguous or does not have the same
        shape.

    Warns
    -----
    PeakPropertyWarning
        Raised if any calculated width is 0. This may stem from the supplied
        `prominence_data` or if `rel_height` is set to 0.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_prominences
        Calculate the prominence of peaks.

    Notes
    -----
    The basic algorithm to calculate a peak's width is as follows:

    * Calculate the evaluation height :math:`h_{eval}` with the formula
      :math:`h_{eval} = h_{Peak} - P \\cdot R`, where :math:`h_{Peak}` is the
      height of the peak itself, :math:`P` is the peak's prominence and
      :math:`R` a positive ratio specified with the argument `rel_height`.
    * Draw a horizontal line at the evaluation height to both sides, starting
      at the peak's current vertical position until the lines either intersect
      a slope, the signal border or cross the vertical position of the peak's
      base (see `peak_prominences` for an definition). For the first case,
      intersection with the signal, the true intersection point is estimated
      with linear interpolation.
    * Calculate the width as the horizontal distance between the chosen
      endpoints on both sides. As a consequence of this the maximal possible
      width for each peak is the horizontal distance between its bases.

    As shown above to calculate a peak's width its prominence and bases must be
    known. You can supply these yourself with the argument `prominence_data`.
    Otherwise, they are internally calculated (see `peak_prominences`).
    """
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    if prominence_data is None:
        # Calculate prominence if not supplied and use wlen if supplied.
        wlen = _arg_wlen_as_expected(wlen)
        prominence_data = _peak_prominences(x, peaks, wlen, check=True)
    return _peak_widths(x, peaks, rel_height, *prominence_data, check=True)


def find_peaks(x, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
    """
    Find peaks inside a signal based on peak properties.

    This function takes a 1-D array and finds all local maxima by
    simple comparison of neighboring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the  minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if
        `width` is given. See argument  `rel_height` in `peak_widths` for
        a full description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.

        .. versionadded:: 1.2.0

    Returns
    -------
    peaks : ndarray
        Indices of peaks in `x` that satisfy all given conditions.
    properties : dict
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * 'peak_heights'
              If `height` is given, the height of each peak in `x`.
        * 'left_thresholds', 'right_thresholds'
              If `threshold` is given, these keys contain a peaks vertical
              distance to its neighbouring samples.
        * 'prominences', 'right_bases', 'left_bases'
              If `prominence` is given, these keys are accessible. See
              `peak_prominences` for a description of their content.
        * 'width_heights', 'left_ips', 'right_ips'
              If `width` is given, these keys are accessible. See `peak_widths`
              for a description of their content.
        * 'plateau_sizes', left_edges', 'right_edges'
              If `plateau_size` is given, these keys are accessible and contain
              the indices of a peak's edges (edges are still part of the
              plateau) and the calculated plateau sizes.

        To calculate and return properties without excluding peaks, provide the
        open interval ``(None, None)`` as a value to the appropriate argument
        (excluding `distance`).

    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths`).

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks_cwt
        Find peaks using the wavelet transformation.
    peak_prominences
        Directly calculate the prominence of peaks.
    peak_widths
        Directly calculate the width of peaks.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `find_peaks_cwt`).

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima
      (see `peak_prominences`).
    """  # NOQA

    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')

    peaks, left_edges, right_edges = _local_maxima_1d(x)
    properties = {}

    if plateau_size is not None:
        # Evaluate plateau size
        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties["plateau_sizes"] = plateau_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        properties = {key: array[keep] for key, array in properties.items()}

    if height is not None:
        # Evaluate height condition
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties["peak_heights"] = peak_heights
        properties = {key: array[keep] for key, array in properties.items()}

    if threshold is not None:
        # Evaluate threshold condition
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, peaks, tmin, tmax)
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        properties = {key: array[keep] for key, array in properties.items()}

    if distance is not None:
        # Evaluate distance condition
        keep = _select_by_peak_distance(peaks, x[peaks], distance)  # NOQA
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:
        # Calculate prominence (required for both conditions)
        wlen = _arg_wlen_as_expected(wlen)  # NOQA
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences(x, peaks, wlen=wlen)  # NOQA
        ))

    if prominence is not None:
        # Evaluate prominence condition
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)  # NOQA
        keep = _select_by_property(properties['prominences'], pmin, pmax)  # NOQA
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:
        # Calculate widths
        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths(x, peaks, rel_height, properties['prominences'],  # NOQA
                         properties['left_bases'], properties['right_bases'])
        ))
        # Evaluate width condition
        wmin, wmax = _unpack_condition_args(width, x, peaks)  # NOQA
        keep = _select_by_property(properties['widths'], wmin, wmax)  # NOQA
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    return peaks, properties


def _peak_finding(data, comparator, axis, order, mode, results):
    comp = _modedict[comparator]
    clip = mode == 'clip'

    device_id = cupy.cuda.Device()
    num_blocks = (device_id.attributes["MultiProcessorCount"] * 20,)
    block_sz = (512,)
    call_args = data.shape[axis], order, clip, comp, data, results

    kernel_name = "boolrelextrema_1D"
    if data.ndim > 1:
        kernel_name = "boolrelextrema_2D"
        block_sz_x, block_sz_y = 16, 16
        n_blocks_x = (data.shape[1] + block_sz_x - 1) // block_sz_x
        n_blocks_y = (data.shape[0] + block_sz_y - 1) // block_sz_y
        block_sz = (block_sz_x, block_sz_y)
        num_blocks = (n_blocks_x, n_blocks_y)
        call_args = (data.shape[1], data.shape[0], order, clip, comp, axis,
                     data, results)

    boolrelextrema = _get_module_func(ARGREL_MODULE, kernel_name, data)
    boolrelextrema(num_blocks, block_sz, call_args)


def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See cupy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin
    """
    if (int(order) != order) or (order < 1):
        raise ValueError("Order must be an int >= 1")

    if data.ndim < 3:
        results = cupy.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        datalen = data.shape[axis]
        locs = cupy.arange(0, datalen)
        results = cupy.ones(data.shape, dtype=bool)
        main = cupy.take(data, locs, axis=axis)
        for shift in cupy.arange(1, order + 1):
            if mode == "clip":
                p_locs = cupy.clip(locs + shift, a_min=None,
                                   a_max=(datalen - 1))
                m_locs = cupy.clip(locs - shift, a_min=0, a_max=None)
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cupy.take(data, p_locs, axis=axis)
            minus = cupy.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)

            if ~results.any():
                return results

    return results


def argrelmin(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative minima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative minima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See cupy.take.


    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the minima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmax, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cupy.less as comparator. Therefore
    it requires a strict inequality on both sides of a value to consider it a
    minimum. This means flat minima (more than one sample wide) are not
    detected. In case of one-dimensional `data` `find_peaks` can be used to
    detect all local minima, including flat ones, by calling it with negated
    `data`.

    Examples
    --------
    >>> from cupyx.scipy.signal import argrelmin
    >>> import cupy
    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = cupy.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cupy.asarray(data)
    return argrelextrema(data, cupy.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See cupy.take.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmin, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cupy.greater as comparator.
    Therefore it requires a strict inequality on both sides of a value to
    consider it a maximum. This means flat maxima (more than one sample wide)
    are not detected. In case of one-dimensional `data` `find_peaks` can be
    used to detect all local maxima, including flat ones.

    Examples
    --------
    >>> from cupyx.scipy.signal import argrelmax
    >>> import cupy
    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = cupy.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    data = cupy.asarray(data)
    return argrelextrema(data, cupy.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See cupy.take.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> from cupyx.scipy.signal import argrelextrema
    >>> import cupy
    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cupy.greater)
    (array([3, 6]),)
    >>> y = cupy.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cupy.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cupy.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)

    if mode == "raise":
        raise NotImplementedError(
            "CuPy `take` doesn't support `mode='raise'`.")

    return cupy.nonzero(results)
