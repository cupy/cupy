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
from cupyx.scipy.signal import windows


def _pulse_preprocess(x, normalize, window):
    if window is not None:
        n = x.shape[-1]
        if callable(window):
            w = window(cupy.fft.fftfreq(n).astype(x.dtype))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (n,):
                raise ValueError("window must have the same length as data")
            w = window
        else:
            w = windows.get_window(window, n, False).astype(x.dtype)
        x = x * w

    if normalize:
        x = x / cupy.linalg.norm(x)

    return x


def pulse_compression(x, template, normalize=False, window=None, nfft=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    num_pulses, samples_per_pulse = x.shape
    dtype = cupy.result_type(x, template)

    if nfft is None:
        nfft = samples_per_pulse

    t = _pulse_preprocess(template, normalize, window)
    fft_x = cupy.fft.fft(x, nfft)
    fft_t = cupy.fft.fft(t, nfft)
    out = cupy.fft.ifft(fft_x * fft_t.conj(), nfft)
    if dtype.kind != 'c':
        out = out.real
    return out


def pulse_doppler(x, window=None, nfft=None):
    """
    Pulse doppler processing yields a range/doppler data matrix that represents
    moving target data that's separated from clutter. An estimation of the
    doppler shift can also be obtained from pulse doppler processing. FFT taken
    across slow-time (pulse) dimension.

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    pd_dataMatrix : ndarray
        Pulse-doppler output (range/doppler matrix)
    """
    num_pulses, samples_per_pulse = x.shape

    if nfft is None:
        nfft = num_pulses

    xT = _pulse_preprocess(x.T, False, window)
    return cupy.fft.fft(xT, nfft).T


def cfar_alpha(pfa, N):
    """
    Computes the value of alpha corresponding to a given probability
    of false alarm and number of reference cells N.

    Parameters
    ----------
    pfa : float
        Probability of false alarm.

    N : int
        Number of reference cells.

    Returns
    -------
    alpha : float
        Alpha value.
    """
    return N * (pfa ** (-1.0 / N) - 1)


def ca_cfar(array, guard_cells, reference_cells, pfa=1e-3):
    """
    Computes the cell-averaged constant false alarm rate (CA CFAR) detector
    threshold and returns for a given array.
    Parameters
    ----------
    array : ndarray
        Array containing data to be processed.
    guard_cells_x : int
        One-sided guard cell count in the first dimension.
    guard_cells_y : int
        One-sided guard cell count in the second dimension.
    reference_cells_x : int
        one-sided reference cell count in the first dimension.
    reference_cells_y : int
        one-sided reference cell count in the second dimension.
    pfa : float
        Probability of false alarm.
    Returns
    -------
    threshold : ndarray
        CFAR threshold
    return : ndarray
        CFAR detections
    """
    shape = array.shape
    if len(shape) > 2:
        raise TypeError('Only 1D and 2D arrays are currently supported.')
    mask = cupy.zeros(shape, dtype=cupy.float32)

    if len(shape) == 1:
        if len(array) <= 2 * guard_cells + 2 * reference_cells:
            raise ValueError('Array too small for given parameters')
        intermediate = cupy.cumsum(array, axis=0, dtype=cupy.float32)
        N = 2 * reference_cells
        alpha = cfar_alpha(pfa, N)
        tpb = (32,)
        bpg = ((len(array) - 2 * reference_cells - 2 * guard_cells +
               tpb[0] - 1) // tpb[0],)
        _ca_cfar_1d_kernel(bpg, tpb, (array, intermediate, mask,
                                      len(array), N, cupy.float32(alpha),
                                      guard_cells, reference_cells))
    elif len(shape) == 2:
        if len(guard_cells) != 2 or len(reference_cells) != 2:
            raise TypeError('Guard and reference cells must be two '
                            'dimensional.')
        guard_cells_x, guard_cells_y = guard_cells
        reference_cells_x, reference_cells_y = reference_cells
        if shape[0] - 2 * guard_cells_x - 2 * reference_cells_x <= 0:
            raise ValueError('Array first dimension too small for given '
                             'parameters.')
        if shape[1] - 2 * guard_cells_y - 2 * reference_cells_y <= 0:
            raise ValueError('Array second dimension too small for given '
                             'parameters.')
        intermediate = cupy.cumsum(array, axis=0, dtype=cupy.float32)
        intermediate = cupy.cumsum(intermediate, axis=1, dtype=cupy.float32)
        N = 2 * reference_cells_x * (2 * reference_cells_y +
                                     2 * guard_cells_y + 1)
        N += 2 * (2 * guard_cells_x + 1) * reference_cells_y
        alpha = cfar_alpha(pfa, N)
        tpb = (8, 8)
        bpg_x = (shape[0] - 2 * (reference_cells_x + guard_cells_x) + tpb[0] -
                 1) // tpb[0]
        bpg_y = (shape[1] - 2 * (reference_cells_y + guard_cells_y) + tpb[1] -
                 1) // tpb[1]
        bpg = (bpg_x, bpg_y)
        _ca_cfar_2d_kernel(bpg, tpb, (array, intermediate, mask,
                           shape[0], shape[1], N, cupy.float32(alpha),
                           guard_cells_x, guard_cells_y, reference_cells_x,
                           reference_cells_y))
    return (mask, array - mask > 0)


_ca_cfar_2d_kernel = cupy.RawKernel(r'''
extern "C" __global__ void
_ca_cfar_2d_kernel(float * array, float * intermediate, float * mask,
                   int width, int height, int N, float alpha,
                   int guard_cells_x, int guard_cells_y,
                   int reference_cells_x, int reference_cells_y)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int j_init = threadIdx.y+blockIdx.y*blockDim.y;
    int i, j, x, y, offset;
    int tro, tlo, blo, bro, tri, tli, bli, bri;
    float outer_area, inner_area, T;
    for (i=i_init; i<width-2*(guard_cells_x+reference_cells_x);
         i += blockDim.x*gridDim.x){
        for (j=j_init; j<height-2*(guard_cells_y+reference_cells_y);
             j += blockDim.y*gridDim.y){
            /* 'tri' is Top Right Inner (square), 'blo' is Bottom Left
             * Outer (square), etc. These are the corners at which
             * the intermediate array must be evaluated.
             */
            x = i+guard_cells_x+reference_cells_x;
            y = j+guard_cells_y+reference_cells_y;
            offset = x*height+y;
            tro = (x+guard_cells_x+reference_cells_x)*height+y+
                guard_cells_y+reference_cells_y;
            tlo = (x-guard_cells_x-reference_cells_x-1)*height+y+
                guard_cells_y+reference_cells_y;
            blo = (x-guard_cells_x-reference_cells_x-1)*height+y-
                guard_cells_y-reference_cells_y-1;
            bro = (x+guard_cells_x+reference_cells_x)*height+y-
                guard_cells_y-reference_cells_y-1;
            tri = (x+guard_cells_x)*height+y+guard_cells_y;
            tli = (x-guard_cells_x-1)*height+y+guard_cells_y;
            bli = (x-guard_cells_x-1)*height+y-guard_cells_y-1;
            bri = (x+guard_cells_x)*height+y-guard_cells_y-1;
            /* It would be nice to eliminate the triple
             * branching here, but it only occurs on the boundaries
             * of the array (i==0 or j==0). So it shouldn't hurt
             * overall performance much.
             */
            if (i>0 && j>0){
                outer_area = intermediate[tro]-intermediate[tlo]-
                    intermediate[bro]+intermediate[blo];
            } else if (i == 0 && j > 0){
                outer_area = intermediate[tro]-intermediate[bro];
            } else if (i > 0 && j == 0){
                outer_area = intermediate[tro]-intermediate[tlo];
            } else if (i == 0 && j == 0){
                outer_area = intermediate[tro];
            }
            inner_area = intermediate[tri]-intermediate[tli]-
                intermediate[bri]+intermediate[bli];
            T = outer_area-inner_area;
            T = alpha/N*T;
            mask[offset] = T;
        }
    }
}
''', '_ca_cfar_2d_kernel')


_ca_cfar_1d_kernel = cupy.RawKernel(r'''
extern "C" __global__ void
_ca_cfar_1d_kernel(float * array, float * intermediate, float * mask,
                   int width, int N, float alpha,
                   int guard_cells, int reference_cells)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int i, x;
    int br, bl, sr, sl;
    float big_area, small_area, T;
    for (i=i_init; i<width-2*(guard_cells+reference_cells);
         i += blockDim.x*gridDim.x){
        x = i+guard_cells+reference_cells;
        br = x+guard_cells+reference_cells;
        bl = x-guard_cells-reference_cells-1;
        sr = x+guard_cells;
        sl = x-guard_cells-1;
        if (i>0){
            big_area = intermediate[br]-intermediate[bl];
        } else{
            big_area = intermediate[br];
        }
        small_area = intermediate[sr]-intermediate[sl];
        T = big_area-small_area;
        T = alpha/N*T;
        mask[x] = T;
    }
}
''', '_ca_cfar_1d_kernel')
