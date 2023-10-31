"""
upfirdn implementation.

Functions defined here were ported directly from cuSignal under
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

from math import ceil
import cupy

_upfirdn_modes = [
    'constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect',
    'antisymmetric', 'antireflect', 'line',
]


UPFIRDN_KERNEL = r'''
#include <cupy/complex.cuh>

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN1D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn1D( const T *__restrict__ inp,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW ) {

    const int t { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( size_t tid = t; tid < outW; tid += stride ) {

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
        __builtin_assume( padded_len > 0 );
        __builtin_assume( up > 0 );
        __builtin_assume( down > 0 );
        __builtin_assume( tid > 0 );
#endif

        const int x_idx { static_cast<int>( ( tid * down ) / up ) % padded_len };
        int       h_idx { static_cast<int>( ( tid * down ) % up * h_per_phase ) };
        int       x_conv_idx { x_idx - h_per_phase + 1 };

        if ( x_conv_idx < 0 ) {
            h_idx -= x_conv_idx;
            x_conv_idx = 0;
        }

        T temp {};

        int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

        for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
            temp += inp[x_c] * h_trans_flip[h_idx];
            h_idx += 1;
        }
        out[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float32( const float *__restrict__ inp,
                                                                             const float *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             float *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<float>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float64( const double *__restrict__ inp,
                                                                             const double *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             double *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<double>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW ) {
    _cupy_upfirdn1D<thrust::complex<float>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW ) {
    _cupy_upfirdn1D<thrust::complex<double>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN2D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn2D( const T *__restrict__ inp,
                                 const int inpH,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW,
                                 const int outH ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    const int stride_y { static_cast<int>( blockDim.x * gridDim.x ) };
    const int stride_x { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int x = tx; x < outH; x += stride_x ) {
        for ( int y = ty; y < outW; y += stride_y ) {
            int x_idx {};
            int h_idx {};

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
            __builtin_assume( padded_len > 0 );
            __builtin_assume( up > 0 );
            __builtin_assume( down > 0 );
#endif

            if ( axis == 1 ) {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( x > 0 );
#endif
                x_idx = ( static_cast<int>( x * down ) / up ) % padded_len;
                h_idx = ( x * down ) % up * h_per_phase;
            } else {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( y > 0 );
#endif
                x_idx = ( static_cast<int>( y * down ) / up ) % padded_len;
                h_idx = ( y * down ) % up * h_per_phase;
            }

            int x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            T temp {};

            int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

            for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
                if ( axis == 1 ) {
                    temp += inp[y * inpH + x_c] * h_trans_flip[h_idx];
                } else {
                    temp += inp[x_c * inpH + x] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[y * outH + x] = temp;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 64 ) _cupy_upfirdn2D_float32( const float *__restrict__ inp,
                                                                            const int inpH,
                                                                            const float *__restrict__ h_trans_flip,
                                                                            const int up,
                                                                            const int down,
                                                                            const int axis,
                                                                            const int x_shape_a,
                                                                            const int h_per_phase,
                                                                            const int padded_len,
                                                                            float *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH ) {
    _cupy_upfirdn2D<float>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void _cupy_upfirdn2D_float64( const double *__restrict__ inp,
                                                    const int inpH,
                                                    const double *__restrict__ h_trans_flip,
                                                    const int up,
                                                    const int down,
                                                    const int axis,
                                                    const int x_shape_a,
                                                    const int h_per_phase,
                                                    const int padded_len,
                                                    double *__restrict__ out,
                                                    const int outW,
                                                    const int outH ) {
    _cupy_upfirdn2D<double>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const int inpH,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW,
                               const int outH ) {
    _cupy_upfirdn2D<thrust::complex<float>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const int inpH,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW,
                                const int outH ) {
    _cupy_upfirdn2D<thrust::complex<double>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}
'''  # NOQA


UPFIRDN_MODULE = cupy.RawModule(
    code=UPFIRDN_KERNEL, options=('-std=c++11',),
    name_expressions=[
        '_cupy_upfirdn1D_float32',
        '_cupy_upfirdn1D_float64',
        '_cupy_upfirdn1D_complex64',
        '_cupy_upfirdn1D_complex128',
        '_cupy_upfirdn2D_float32',
        '_cupy_upfirdn2D_float64',
        '_cupy_upfirdn2D_complex64',
        '_cupy_upfirdn2D_complex128',
    ])


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = cupy.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


def _output_len(len_h, in_len, up, down):
    return (((in_len - 1) * up + len_h) - 1) // down + 1


# These three _get_* functions are vendored from
# https://github.com/rapidsai/cusignal/blob/branch-23.08/python/cusignal/utils/helper_tools.py#L55
def _get_max_gdx():
    device_id = cupy.cuda.Device()
    return device_id.attributes["MaxGridDimX"]


def _get_max_gdy():
    device_id = cupy.cuda.Device()
    return device_id.attributes["MaxGridDimY"]


def _get_tpb_bpg():
    device_id = cupy.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cupy.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = cupy.result_type(h.dtype, x_dtype, cupy.float32)
        h = cupy.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cupy.asarray(self._h_trans_flip)
        self._h_trans_flip = cupy.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(
        self,
        x,
        axis,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        x = cupy.asarray(x, self._output_type)

        output_len = _output_len(
            self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cupy.empty(output_shape, dtype=self._output_type, order="C")
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim == 1:

            threadsperblock, blockspergrid = _get_tpb_bpg()

            kernel = UPFIRDN_MODULE.get_function(
                f'_cupy_upfirdn1D_{out.dtype.name}')
            kernel(((x.shape[0] + 128 - 1) // 128,), (128,),
                   (x,
                    self._h_trans_flip,
                    self._up,
                    self._down,
                    axis,
                    x_shape_a,
                    h_per_phase,
                    padded_len,
                    out,
                    out.shape[0]
                    )
                   )

        elif out.ndim == 2:
            # set up the kernel launch parameters
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = (
                blocks if blocks < _get_max_gdx() else _get_max_gdx())

            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = (
                blocks if blocks < _get_max_gdy() else _get_max_gdy())

            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # do computations
            kernel = UPFIRDN_MODULE.get_function(
                f'_cupy_upfirdn2D_{out.dtype.name}')
            kernel(threadsperblock, blockspergrid,
                   (x,
                    x.shape[1],
                    self._h_trans_flip,
                    self._up,
                    self._down,
                    axis,
                    x_shape_a,
                    h_per_phase,
                    padded_len,
                    out,
                    out.shape[0],
                    out.shape[1]
                    )
                   )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        return out


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    mode=None,
    cval=0
):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
    mode : str, optional
        This parameter is not implemented.
    cval : float, optional
        This parameter is not implemented.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    See Also
    --------
    scipy.signal.upfirdn

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """
    if mode is not None or cval != 0:
        raise NotImplementedError(f"{mode = } and {cval =} not implemented.")

    ufd = _UpFIRDn(h, x.dtype, int(up), int(down))
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
