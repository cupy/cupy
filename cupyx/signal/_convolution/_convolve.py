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

import cupy


_convolve1d2o_kernel = cupy.ElementwiseKernel(
    'raw T in1, raw T in2, int32 W, int32 H', 'T out',
    """
    T temp {};
    for (int x = 0; x < W; x++) {
      for (int y = 0; y < H; y++) {
        temp += in1[i + W - x - 1] * in1[i + H - y - 1] * in2[H * x + y];
      }
    }
    out = temp;
    """,
    "cupy_convolved2o",
)


def _convolve1d2o(in1, in2, mode):
    assert mode == "valid"
    out_dim = in1.shape[0] - max(in2.shape) + 1
    dtype = cupy.result_type(in1, in2)
    out = cupy.empty(out_dim, dtype=dtype)
    _convolve1d2o_kernel(in1, in2, *in2.shape, out)
    return out


def convolve1d2o(in1, in2, mode='valid', method='direct'):
    """
    Convolve a 1-dimensional arrays with a 2nd order filter.
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

    Examples
    --------
    Convolution of a 2nd order filter on a 1d signal

    >>> import cusignal as cs
    >>> import numpy as np
    >>> d = 50
    >>> a = np.random.uniform(-1,1,(200))
    >>> b = np.random.uniform(-1,1,(d,d))
    >>> c = cs.convolve1d2o(a,b)

    """

    if in1.ndim != 1:
        raise ValueError('in1 should have one dimension')
    if in2.ndim != 2:
        raise ValueError('in2 should have three dimension')

    if mode in ["same", "full"]:
        raise NotImplementedError("Mode == {} not implemented".format(mode))

    if method == "direct":
        return _convolve1d2o(in1, in2, mode)
    else:
        raise NotImplementedError("Only Direct method implemented")


_convolve1d3o_kernel = cupy.ElementwiseKernel(
    'raw T in1, raw T in2, int32 W, int32 H, int32 D', 'T out',
    """
    T temp {};
    for (int x = 0; x < W; x++) {
      for (int y = 0; y < H; y++) {
        for (int z = 0; z < D; z++) {
          temp += in1[i + W - x - 1] * in1[i + H - y - 1] *
                  in1[i + D - z - 1] * in2[(H * x + y) * D + z];
        }
      }
    }
    out = temp;
    """,
    "cupy_convolved3o",
)


def _convolve1d3o(in1, in2, mode):
    assert mode == "valid"
    out_dim = in1.shape[0] - max(in2.shape) + 1
    dtype = cupy.result_type(in1, in2)
    out = cupy.empty(out_dim, dtype=dtype)
    _convolve1d3o_kernel(in1, in2, *in2.shape, out)
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
