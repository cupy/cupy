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
from __future__ import annotations


import cupy
from cupy._core._scalar import get_typename
from cupyx.scipy._lib._util import _get_index_type

_upfirdn_modes = [
    'constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect',
    'antisymmetric', 'antireflect', 'line',
]


UPFIRDN_KERNEL = r'''
#include <cupy/complex.cuh>

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN1D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T, typename index_t>
__global__ void _cupy_upfirdn1D( const T *__restrict__ inp,
                                 const long long inp_stride,
                                 const T *__restrict__ h_trans_flip,
                                 const index_t up,
                                 const index_t down,
                                 const int axis,
                                 const index_t x_shape_a,
                                 const index_t h_per_phase,
                                 const index_t padded_len,
                                 T *__restrict__ out,
                                 const index_t outW ) {

    const index_t t { index_t(blockIdx.x) * blockDim.x + threadIdx.x };
    const index_t stride { index_t(blockDim.x) * gridDim.x };

    for ( index_t tid = t; tid < outW; tid += stride ) {

        __builtin_assume( padded_len > 0 );
        __builtin_assume( up > 0 );
        __builtin_assume( down > 0 );

        const index_t x_idx { static_cast<index_t>( ( tid * down ) / up ) % padded_len };
        index_t       h_idx { static_cast<index_t>( ( tid * down ) % up * h_per_phase ) };
        index_t       x_conv_idx { x_idx - h_per_phase + 1 };

        if ( x_conv_idx < 0 ) {
            h_idx -= x_conv_idx;
            x_conv_idx = 0;
        }

        T temp {};

        index_t stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

        for ( index_t x_c = x_conv_idx; x_c < stop; x_c++ ) {
            temp += inp[x_c * inp_stride] * h_trans_flip[h_idx];
            h_idx += 1;
        }
        out[tid] = temp;
    }
}

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN2D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T, typename index_t>
__global__ void _cupy_upfirdn2D( const T *__restrict__ inp,
                                 const long long inp_strideW,
                                 const long long inp_strideH,
                                 const T *__restrict__ h_trans_flip,
                                 const index_t up,
                                 const index_t down,
                                 const int axis,
                                 const index_t x_shape_a,
                                 const index_t h_per_phase,
                                 const index_t padded_len,
                                 T *__restrict__ out,
                                 const index_t outW,
                                 const index_t outH ) {

    const index_t ty { index_t(blockIdx.x) * blockDim.x + threadIdx.x };
    const index_t tx { index_t(blockIdx.y) * blockDim.y + threadIdx.y };

    const index_t stride_y { index_t(blockDim.x) * gridDim.x };
    const index_t stride_x { index_t(blockDim.y) * gridDim.y };

    for ( index_t x = tx; x < outH; x += stride_x ) {
        for ( index_t y = ty; y < outW; y += stride_y ) {
            index_t x_idx {};
            index_t h_idx {};

            __builtin_assume( padded_len > 0 );
            __builtin_assume( up > 0 );
            __builtin_assume( down > 0 );

            if ( axis == 1 ) {
                x_idx = ( static_cast<index_t>( x * down ) / up ) % padded_len;
                h_idx = ( x * down ) % up * h_per_phase;
            } else {
                x_idx = ( static_cast<index_t>( y * down ) / up ) % padded_len;
                h_idx = ( y * down ) % up * h_per_phase;
            }

            index_t x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            T temp {};

            index_t stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

            for ( index_t x_c = x_conv_idx; x_c < stop; x_c++ ) {
                if ( axis == 1 ) {
                    temp += inp[y * inp_strideH + x_c * inp_strideW] * h_trans_flip[h_idx];
                } else {
                    temp += inp[x_c * inp_strideH + x * inp_strideW] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[y * outH + x] = temp;
        }
    }
}


'''  # NOQA


UPFIRDN_MODULE = cupy.RawModule(
    code=UPFIRDN_KERNEL,
    name_expressions=[
        f'_cupy_upfirdn{ndim}D<{t}, {i}>'
        for ndim in (1, 2)
        for t in ('float', 'double', 'thrust::complex<float>',
                  'thrust::complex<double>')
        for i in ('int', 'long long')])


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


class _UpFIRDn:
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
        # Element strides require byte strides divisible by itemsize.
        # CuPy arrays always satisfy this in practice.
        assert all(s % x.itemsize == 0 for s in x.strides)

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

            inp_stride = x.strides[axis] // x.itemsize

            index_type = _get_index_type(
                x, self._h_trans_flip, out,
                max_size=max(padded_len, out.size * self._down))
            kernel = UPFIRDN_MODULE.get_function(
                f'_cupy_upfirdn1D<{get_typename(out.dtype)}, {index_type}>')
            kernel((min((out.size + 127) // 128, 65535),), (128,),
                   (x,
                    inp_stride,
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
            self._apply_filter_2d(x, out, axis=axis,
                                  x_shape_a=x_shape_a,
                                  h_per_phase=h_per_phase,
                                  padded_len=padded_len)
        else:
            # N-D case: reshape to 2D, apply filter, reshape back

            # Move target axis to the end
            x_moved = cupy.ascontiguousarray(cupy.moveaxis(x, axis, -1))
            shape_with_axis_at_end = x_moved.shape

            # Reshape to 2d
            x_2d = x_moved.reshape(-1, x_moved.shape[-1])
            out_2d = cupy.empty((x_2d.shape[0], output_len),
                                dtype=self._output_type, order="C")
            self._apply_filter_2d(x_2d, out_2d, axis=1,
                                  x_shape_a=x_shape_a,
                                  h_per_phase=h_per_phase,
                                  padded_len=padded_len)

            # Reshape back to N-D
            new_shape = shape_with_axis_at_end[:-1] + (output_len,)
            out_nd = out_2d.reshape(new_shape)
            out = cupy.ascontiguousarray(cupy.moveaxis(out_nd, -1, axis))

        return out

    def _apply_filter_2d(self, x, out, axis,
                         x_shape_a, h_per_phase,
                         padded_len):
        """Apply the 2D upfirdn kernel."""
        # set up the kernel launch parameters
        threadsperblock = (8, 8)

        blocks = (out.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_x = min(blocks, _get_max_gdx())

        blocks = (out.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid_y = min(blocks, _get_max_gdy())

        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # do computations
        inp_strideW = x.strides[1] // x.itemsize
        inp_strideH = x.strides[0] // x.itemsize
        index_type = _get_index_type(
            x, self._h_trans_flip, out,
            max_size=max(padded_len, out.shape[axis] * self._down))
        kernel = UPFIRDN_MODULE.get_function(
            f'_cupy_upfirdn2D<{get_typename(out.dtype)}, {index_type}>')
        kernel(blockspergrid, threadsperblock,
               (x,
                inp_strideW,
                inp_strideH,
                self._h_trans_flip,
                self._up,
                self._down,
                axis,
                x_shape_a,
                h_per_phase,
                padded_len,
                out,
                out.shape[0],
                out.shape[1]))


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    mode="constant",
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
        This parameter is not implemented for values other than ``"constant"``.
    cval : float, optional
        This parameter is not implemented for values other than 0.

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
    if mode is None:
        mode = "constant"  # For backwards compatibility
    if mode != "constant" or cval != 0:
        raise NotImplementedError(f"{mode=} and {cval=} not implemented.")

    ufd = _UpFIRDn(h, x.dtype, int(up), int(down))
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
