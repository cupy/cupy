"""
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

import numpy as np

import cupy as cp
from cupy._core._scalar import get_typename
from cupyx.signal.convolution import _convolution_utils


CONVOLVE1D3O_KERNEL = """
#include <cupy/complex.cuh>

///////////////////////////////////////////////////////////////////////////////
//                              CONVOLVE 1D3O                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void _cupy_convolve1D3O( const T *__restrict__ inp,
                                const int inpW,
                                const T *__restrict__ kernel,
                                const int  kerW,
                                const int  kerH,
                                const int  kerD,
                                const int  mode,
                                T *__restrict__ out,
                                const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {

        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int i = 0; i < kerW; i++ ) {
                    for ( int j = 0; j < kerH; j++ ) {
                        for ( int k = 0; k < kerD; k++ ) {
                            temp += inp[tid + kerW - i - 1] * inp[tid + kerH - j - 1] * inp[tid + kerD - k - 1] * kernel[ (kerW * i + j) * kerH + k ];
                        }
                    }
                }
            }
        }
        out[tid] = temp;
    }

}
"""  # NOQA

CONVOLVE1D3O_MODULE = cp.RawModule(
    code=CONVOLVE1D3O_KERNEL, options=('-std=c++11',),
    name_expressions=[
        '_cupy_convolve1D3O<float>',
        '_cupy_convolve1D3O<double>',
        '_cupy_convolve1D3O<complex<float>>',
        '_cupy_convolve1D3O<complex<double>>',
    ])


def _convolve1d3o_gpu(inp, out, ker, mode):

    kernel = CONVOLVE1D3O_MODULE.get_function(
        f'_cupy_convolve1D3O<{get_typename(out.dtype)}>')

    threadsperblock = (out.shape[0] + 128 - 1) // 128,
    blockspergrid = 128,
    kernel_args = (
        inp,
        inp.shape[0],
        ker,
        *ker.shape,
        mode,
        out,
        out.shape[0],
    )
    kernel(threadsperblock, blockspergrid, kernel_args)

    return out


def _convolve1d3o(in1, in2, mode):

    val = _convolution_utils._valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, int)
    if val == _convolution_utils.VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve1d3o_gpu(in1, out, in2, val)

    return out


def convolve1d3o(in1, in2, mode='valid', method='direct'):
    """
    Convolve a 1-dimensional array with a 3rd order filter.
    This results in a second order convolution.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    out : ndarray
        A 1-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve
    convolve1d2o
    convolve1d3o
    """

    signal = in1
    kernel = in2
    if mode == "valid" and signal.shape[0] < kernel.shape[0]:
        # Convolution is commutative
        # order doesn't have any effect on output
        signal, kernel = kernel, signal

    if mode in ["same", "full"]:
        raise NotImplementedError("Mode == {} not implemented".format(mode))

    if method == "direct":
        return _convolve1d3o(signal, kernel, mode)
    else:
        raise NotImplementedError("Only Direct method implemented")
