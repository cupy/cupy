"""
Spectral analysis functions and utilities.

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

import warnings

import cupy

import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
    odd_ext, even_ext, zero_ext, const_ext, _as_strided)
from cupyx.scipy.signal.windows._windows import get_window


def _get_raw_typename(dtype):
    return cupy.dtype(dtype).name


def _get_module_func_raw(module, func_name, *template_args):
    args_dtypes = [_get_raw_typename(arg.dtype) for arg in template_args]
    template = '_'.join(args_dtypes)
    kernel_name = f'{func_name}_{template}' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


LOMBSCARGLE_KERNEL = r"""

///////////////////////////////////////////////////////////////////////////////
//                            LOMBSCARGLE                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_lombscargle_float( const int x_shape,
                                         const int freqs_shape,
                                         const T *__restrict__ x,
                                         const T *__restrict__ y,
                                         const T *__restrict__ freqs,
                                         T *__restrict__ pgram,
                                         const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0f;
    } else {
        yD = 2.0f / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {
            sincosf( freq * x[j], &s, &c );
            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T c_tau {};
        T s_tau {};
        T tau { atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
        sincosf( freq * tau, &s_tau, &c_tau );
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0f * c_tau * s_tau };

        pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) *
                                  ( c_tau * xc + s_tau * xs ) /
                                  ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                                ( ( c_tau * xs - s_tau * xc ) *
                                  ( c_tau * xs - s_tau * xc ) /
                                  ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float32(
        const int x_shape, const int freqs_shape, const float *__restrict__ x,
        const float *__restrict__ y, const float *__restrict__ freqs,
        float *__restrict__ pgram, const float *__restrict__ y_dot ) {
    _cupy_lombscargle_float<float>( x_shape, freqs_shape, x, y,
                                    freqs, pgram, y_dot );
}

template<typename T>
__device__ void _cupy_lombscargle_double( const int x_shape,
                                          const int freqs_shape,
                                          const T *__restrict__ x,
                                          const T *__restrict__ y,
                                          const T *__restrict__ freqs,
                                          T *__restrict__ pgram,
                                          const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0;
    } else {
        yD = 2.0 / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {

            sincos( freq * x[j], &s, &c );
            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T c_tau {};
        T s_tau {};
        T tau { atan2( 2.0 * cs, cc - ss ) / ( 2.0 * freq ) };
        sincos( freq * tau, &s_tau, &c_tau );
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0 * c_tau * s_tau };

        pgram[tid] = ( 0.5 * ( ( ( c_tau * xc + s_tau * xs ) *
                                 ( c_tau * xc + s_tau * xs ) /
                                 ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                               ( ( c_tau * xs - s_tau * xc ) *
                                 ( c_tau * xs - s_tau * xc ) /
                                 ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float64(
        const int x_shape, const int freqs_shape, const double *__restrict__ x,
        const double *__restrict__ y, const double *__restrict__ freqs,
        double *__restrict__ pgram, const double *__restrict__ y_dot ) {

    _cupy_lombscargle_double<double>( x_shape, freqs_shape, x, y, freqs,
                                      pgram, y_dot );
}
"""  # NOQA


LOMBSCARGLE_MODULE = cupy.RawModule(
    code=LOMBSCARGLE_KERNEL, options=('-std=c++11',),
    name_expressions=['_cupy_lombscargle_float32',
                      '_cupy_lombscargle_float64'])


def _lombscargle(x, y, freqs, pgram, y_dot):
    device_id = cupy.cuda.Device()

    num_blocks = device_id.attributes["MultiProcessorCount"] * 20
    block_sz = 512
    lombscargle_kernel = _get_module_func_raw(
        LOMBSCARGLE_MODULE, '_cupy_lombscargle', x)

    args = (x.shape[0], freqs.shape[0], x, y, freqs, pgram, y_dot)
    lombscargle_kernel((num_blocks,), (block_sz,), args)


def _spectral_helper(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
    boundary=None,
    padded=False,
):
    """
    Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    """
    if mode not in ["psd", "stft"]:
        raise ValueError(
            f"Unknown value for mode {mode}, must be one of: "
            "{'psd', 'stft'}"
        )

    boundary_funcs = {
        "even": even_ext,
        "odd": odd_ext,
        "constant": const_ext,
        "zeros": zero_ext,
        None: None,
    }

    if boundary not in boundary_funcs:
        raise ValueError(
            "Unknown boundary option '{0}', must be one of: {1}".format(
                boundary, list(boundary_funcs.keys())
            )
        )

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != "psd":
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have cp.arrays, get outdtype
    x = cupy.asarray(x)
    if not same_data:
        y = cupy.asarray(y)
        outdtype = cupy.result_type(x, y, cupy.complex64)
    else:
        outdtype = cupy.result_type(x, cupy.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = cupy.broadcast(
                cupy.empty(xouter), cupy.empty(youter)).shape
        except ValueError:
            raise ValueError("x and y cannot be broadcast together.")

    if same_data:
        if x.size == 0:
            return (
                cupy.empty(x.shape), cupy.empty(x.shape), cupy.empty(x.shape))
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = cupy.rollaxis(cupy.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = cupy.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = cupy.rollaxis(y, axis, len(y.shape))

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = cupy.concatenate((x, cupy.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = cupy.concatenate((y, cupy.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = cupy.concatenate((x, cupy.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = cupy.concatenate((y, cupy.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:

        def detrend_func(d):
            return d

    elif not hasattr(detrend, "__call__"):

        def detrend_func(d):
            return filtering.detrend(d, type=detrend, axis=-1)

    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = cupy.rollaxis(d, -1, axis)
            d = detrend(d)
            return cupy.rollaxis(d, axis, len(d.shape))

    else:
        detrend_func = detrend

    if cupy.result_type(win, cupy.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    if mode == "stft":
        scale = cupy.sqrt(scale)

    if return_onesided:
        if cupy.iscomplexobj(x):
            sides = "twosided"
            warnings.warn(
                "Input data is complex, switching to "
                "return_onesided=False"
            )
        else:
            sides = "onesided"
            if not same_data:
                if cupy.iscomplexobj(y):
                    sides = "twosided"
                    warnings.warn(
                        "Input data is complex, switching to "
                        "return_onesided=False"
                    )
    else:
        sides = "twosided"

    if sides == "twosided":
        freqs = cupy.fft.fftfreq(nfft, 1 / fs)
    elif sides == "onesided":
        freqs = cupy.fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func,  # NOQA
                               nperseg, noverlap, nfft, sides)
        result = cupy.conj(result) * result_y
    elif mode == "psd":
        result = cupy.conj(result) * result

    result *= scale
    if sides == "onesided" and mode == "psd":
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = cupy.arange(
        nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap
    ) / float(fs)
    if boundary is not None:
        time -= (nperseg / 2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != "stft":
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = cupy.rollaxis(result, -1, axis)

    return freqs, time, result


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        6
        window.
    """

    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, input_length)
            )
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different"
                    " from length of window"
                )
    return win, nperseg


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    cusignal.spectral_analysis.spectral._spectral_helper

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., cupy.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        # Need to optimize this in cuSignal
        result = _as_strided(x, shape=shape, strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == "twosided":
        func = cupy.fft.fft
    else:
        result = result.real
        func = cupy.fft.rfft
    result = func(result, n=nfft)

    return result


def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See arXiv:gr-qc/0509116 Appendix B for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.
    """
    ii_2 = 2 * cupy.arange(1.0, (n - 1) // 2 + 1)
    return 1 + cupy.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)
