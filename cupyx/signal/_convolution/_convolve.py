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

import cupy as cp
from cupy._core._scalar import get_typename
from cupyx.signal._convolution import _convolution_utils


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
                            temp += inp[tid + kerW - i - 1] * inp[tid + kerH - j - 1] * inp[tid + kerD - k - 1] * kernel[ (kerH * i + j) * kerD + k ];
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


def _convolve1d3o(in1, in2, mode):

    val = _convolution_utils._valfrommode(mode)
    assert val == _convolution_utils.VALID

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    out_dim = in1.shape[0] - max(in2.shape) + 1
    out = cp.empty(out_dim, dtype=in1.dtype)

    _convolve1d3o_gpu(in1, out, in2, val)

    return out


def convolve1d3o(in1, in2, mode='valid', method='direct'):
    """
    Convolve a 1-dimensional array with a 3rd order filter.
    This results in a third order convolution.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input. Should have one dimension.
    in2 : array_like
        Second input. Should have three dimensions.
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

    if in1.ndim != 1:
        raise ValueError('in1 should have one dimension')
    if in2.ndim != 3:
        raise ValueError('in2 should have three dimension')

    if mode in ["same", "full"]:
        raise NotImplementedError("Mode == {} not implemented".format(mode))

    if method == "direct":
        return _convolve1d3o(in1, in2, mode)
    else:
        raise NotImplementedError("Only Direct method implemented")
