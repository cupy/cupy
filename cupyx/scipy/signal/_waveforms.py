
"""
Waveform-generating functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
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
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

import numpy as np


def _get_typename(dtype):
    typename = get_typename(dtype)
    if typename == 'float16':
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
COMPLEX_TYPES = [cupy.complex64, cupy.complex128]
TYPES = FLOAT_TYPES + INT_TYPES + UNSIGNED_TYPES + COMPLEX_TYPES  # type: ignore  # NOQA
TYPE_NAMES = [_get_typename(t) for t in TYPES]


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


_sawtooth_kernel = cupy.ElementwiseKernel(
    "T t, T w",
    "float64 y",
    """
    double out {};
    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };
    if ( mask1 ) {
        out = nan("0xfff8000000000000ULL");
    }

    const T tmod { fmod( t, 2.0 * M_PI ) };
    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };

    if ( mask2 ) {
        out = tmod / ( M_PI * w ) - 1;
    }

    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };
    if ( mask3 ) {
        out = ( M_PI * ( w + 1 ) - tmod ) / ( M_PI * ( 1 - w ) );
    }
    y = out;
    """,
    "_sawtooth_kernel",
    options=("-std=c++11",),
)


def sawtooth(t, width=1.0):
    """
    Return a periodic sawtooth or triangle waveform.

    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        Time.
    width : array_like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the sawtooth waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> from cupyx.scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(0, 1, 500)
    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))
    """
    t, w = cupy.asarray(t), cupy.asarray(width)
    y = _sawtooth_kernel(t, w)
    return y


_square_kernel = cupy.ElementwiseKernel(
    "T t, T w",
    "float64 y",
    """
    const bool mask1 { ( ( w > 1 ) || ( w < 0 ) ) };
    if ( mask1 ) {
        y = nan("0xfff8000000000000ULL");
    }

    const T tmod { fmod( t, 2.0 * M_PI ) };
    const bool mask2 { ( ( 1 - mask1 ) && ( tmod < ( w * 2.0 * M_PI ) ) ) };

    if ( mask2 ) {
        y = 1;
    }

    const bool mask3 { ( ( 1 - mask1 ) && ( 1 - mask2 ) ) };
    if ( mask3 ) {
        y = -1;
    }

    """,
    "_square_kernel",
    options=("-std=c++11",),
)


def square(t, duty=0.5):
    """
    Return a periodic square-wave waveform.

    The square wave has a period ``2*pi``, has value +1 from 0 to
    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
    the interval [0,1].

    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.

    Parameters
    ----------
    t : array_like
        The input time array.
    duty : array_like, optional
        Duty cycle.  Default is 0.5 (50% duty cycle).
        If an array, causes wave shape to change over time, and must be the
        same length as t.

    Returns
    -------
    y : ndarray
        Output array containing the square waveform.

    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:

    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cupy.linspace(0, 1, 500, endpoint=False)
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(cupyx.scipy.signal.square(2 * cupy.pi * 5 * t)))
    >>> plt.ylim(-2, 2)

    A pulse-width modulated sine wave:

    >>> plt.figure()
    >>> sig = cupy.sin(2 * cupy.pi * t)
    >>> pwm = cupyx.scipy.signal.square(2 * cupy.pi * 30 * t, duty=(sig + 1)/2)
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(sig))
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(pwm))
    >>> plt.ylim(-1.5, 1.5)

    """  # NOQA
    t, w = cupy.asarray(t), cupy.asarray(duty)
    y = _square_kernel(t, w)
    return y


_gausspulse_kernel_F_F = cupy.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI",
    """
    T yenv = exp(-a * t * t);
    yI = yenv * cos( 2 * M_PI * fc * t);
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_F_T = cupy.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yenv",
    """
    yenv = exp(-a * t * t);
    yI = yenv * cos( 2 * M_PI * fc * t);
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_T_F = cupy.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yQ",
    """
    T yenv { exp(-a * t * t) };

    T l_yI {};
    T l_yQ {};
    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);
    yI = yenv * l_yI;
    yQ = yenv * l_yQ;
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)

_gausspulse_kernel_T_T = cupy.ElementwiseKernel(
    "T t, T a, T fc",
    "T yI, T yQ, T yenv",
    """
    yenv = exp(-a * t * t);

    T l_yI {};
    T l_yQ {};
    sincos(2 * M_PI * fc * t, &l_yQ, &l_yI);
    yI = yenv * l_yI;
    yQ = yenv * l_yQ;
    """,
    "_gausspulse_kernel",
    options=("-std=c++11",),
)


def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
               retenv=False):
    """
    Return a Gaussian modulated sinusoid:

        ``exp(-a t^2) exp(1j*2*pi*fc*t).``

    If `retquad` is True, then return the real and imaginary parts
    (in-phase and quadrature).
    If `retenv` is True, then return the envelope (unmodulated signal).
    Otherwise, return the real part of the modulated sinusoid.

    Parameters
    ----------
    t : ndarray or the string 'cutoff'
        Input array.
    fc : int, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.
    tpr : float, optional
        If `t` is 'cutoff', then the function returns the cutoff
        time for when the pulse amplitude falls below `tpr` (in dB).
        Default is -60.
    retquad : bool, optional
        If True, return the quadrature (imaginary) as well as the real part
        of the signal.  Default is False.
    retenv : bool, optional
        If True, return the envelope of the signal.  Default is False.

    Returns
    -------
    yI : ndarray
        Real part of signal.  Always returned.
    yQ : ndarray
        Imaginary part of signal.  Only returned if `retquad` is True.
    yenv : ndarray
        Envelope of signal.  Only returned if `retenv` is True.

    See Also
    --------
    cupyx.scipy.signal.morlet

    Examples
    --------
    Plot real component, imaginary component, and envelope for a 5 Hz pulse,
    sampled at 100 Hz for 2 seconds:

    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cupy.linspace(-1, 1, 2 * 100, endpoint=False)
    >>> i, q, e = cupyx.scipy.signal.gausspulse(t, fc=5, retquad=True, retenv=True)
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(i), cupy.asnumpy(t), cupy.asnumpy(q),
                 cupy.asnumpy(t), cupy.asnumpy(e), '--')

    """  # NOQA
    if fc < 0:
        raise ValueError("Center frequency (fc=%.2f) must be >=0." % fc)
    if bw <= 0:
        raise ValueError("Fractional bandwidth (bw=%.2f) must be > 0." % bw)
    if bwr >= 0:
        raise ValueError(
            "Reference level for bandwidth (bwr=%.2f) must " "be < 0 dB" % bwr
        )

    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)

    ref = pow(10.0, bwr / 20.0)
    # fdel = fc*bw/2:  g(fdel) = ref --- solve this for a
    #
    # pi^2/a * fc^2 * bw^2 /4=-log(ref)
    a = -((np.pi * fc * bw) ** 2) / (4.0 * np.log(ref))

    if isinstance(t, str):
        if t == "cutoff":  # compute cut_off point
            #  Solve exp(-a tc**2) = tref  for tc
            #   tc = sqrt(-log(tref) / a) where tref = 10^(tpr/20)
            if tpr >= 0:
                raise ValueError(
                    "Reference level for time cutoff must " "be < 0 dB")
            tref = pow(10.0, tpr / 20.0)
            return np.sqrt(-np.log(tref) / a)
        else:
            raise ValueError("If `t` is a string, it must be 'cutoff'")

    t = cupy.asarray(t)

    if not retquad and not retenv:
        return _gausspulse_kernel_F_F(t, a, fc)
    if not retquad and retenv:
        return _gausspulse_kernel_F_T(t, a, fc)
    if retquad and not retenv:
        return _gausspulse_kernel_T_F(t, a, fc)
    if retquad and retenv:
        return _gausspulse_kernel_T_T(t, a, fc)


_chirp_phase_lin_kernel_real = cupy.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)

_chirp_phase_lin_kernel_cplx = cupy.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "Y phase",
    """
    const T beta { (f1 - f0) / t1 };
    const T temp { 2 * M_PI * (f0 * t + 0.5 * beta * t * t) };
    // Convert  phi to radians.
    phase = Y(cos(temp + phi), cos(temp + phi + M_PI/2) * -1);
    """,
    "_chirp_phase_lin_kernel",
    options=("-std=c++11",),
)

_chirp_phase_quad_kernel = cupy.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi, bool vertex_zero",
    "T phase",
    """
    T temp {};
    const T beta { (f1 - f0) / (t1 * t1) };
    if ( vertex_zero ) {
        temp = 2 * M_PI * (f0 * t + beta * (t * t * t) / 3);
    } else {
        temp = 2 * M_PI *
            ( f1 * t + beta *
            ( ( (t1 - t) * (t1 - t) * (t1 - t) ) - (t1 * t1 * t1)) / 3);
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_quad_kernel",
    options=("-std=c++11",),
)

_chirp_phase_log_kernel = cupy.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T beta { t1 / log(f1 / f0) };
        temp = 2 * M_PI * beta * f0 * ( pow(f1 / f0, t / t1) - 1.0 );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_log_kernel",
    options=("-std=c++11",),
)

_chirp_phase_hyp_kernel = cupy.ElementwiseKernel(
    "T t, T f0, T t1, T f1, T phi",
    "T phase",
    """
    T temp {};
    if ( f0 == f1 ) {
        temp = 2 * M_PI * f0 * t;
    } else {
        T sing { -f1 * t1 / (f0 - f1) };
        temp = 2 * M_PI * ( -sing * f0 ) * log( abs( 1 - t / sing ) );
    }
    // Convert  phi to radians.
    phase = cos(temp + phi);
    """,
    "_chirp_phase_hyp_kernel",
    options=("-std=c++11",),
)


def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True):
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    Examples
    --------
    The following will be used in the examples:

    >>> from cupyx.scipy.signal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt
    >>> import cupy as cp

    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = cupy.linspace(0, 10, 5001)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(w))
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()

    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `cupyx.scipy.signal.spectrogram`.
    We'll use a 10 second interval sampled at 8000 Hz.

    >>> fs = 8000
    >>> T = 10
    >>> t = cupy.linspace(0, T, T*fs, endpoint=False)

    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
    ...                           nfft=2048)
    >>> plt.pcolormesh(cupy.asnumpy(tt), cupy.asnumpy(ff[:513]),
                       cupy.asnumpy(Sxx[:513]), cmap='gray_r')
    >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
    >>> plt.xlabel('t (sec)')
    >>> plt.ylabel('Frequency (Hz)')
    >>> plt.grid()
    >>> plt.show()
    """
    t = cupy.asarray(t)

    if cupy.issubdtype(t.dtype, cupy.integer):
        t = t.astype(cupy.float64)

    phi *= np.pi / 180
    type = 'real'

    if method in ["linear", "lin", "li"]:
        if type == "real":
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        elif type == "complex":
            # type hard-coded to 'real' above, so this code path is never used
            if t.real.dtype.kind == 'f' and t.dtype.itemsize == 8:
                phase = cupy.empty(t.shape, dtype=cupy.complex128)
            else:
                phase = cupy.empty(t.shape, dtype=cupy.complex64)
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            return phase
        else:
            raise NotImplementedError("No kernel for type {}".format(type))

    elif method in ["quadratic", "quad", "q"]:
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)

    elif method in ["logarithmic", "log", "lo"]:
        if f0 * f1 <= 0.0:
            raise ValueError(
                "For a logarithmic chirp, f0 and f1 must be "
                "nonzero and have the same sign."
            )
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)

    elif method in ["hyperbolic", "hyp"]:
        if f0 == 0 or f1 == 0:
            raise ValueError(
                "For a hyperbolic chirp, f0 and f1 must be " "nonzero.")
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)

    else:
        raise ValueError(
            "method must be 'linear', 'quadratic', 'logarithmic',"
            " or 'hyperbolic', but a value of %r was given." % method
        )


UNIT_KERNEL = r'''
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>


template<typename T>
__global__ void unit_impulse(const int n, const int iidx, T* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) {
        return;
    }

    if(idx == iidx) {
        out[idx] = 1;
    } else {
        out[idx] = 0;
    }
}
'''

UNIT_MODULE = cupy.RawModule(
    code=UNIT_KERNEL, options=('-std=c++11',),
    name_expressions=[f'unit_impulse<{x}>' for x in TYPE_NAMES])


def unit_impulse(shape, idx=None, dtype=float):
    """
    Unit impulse signal (discrete delta function) or unit basis vector.

    Parameters
    ----------
    shape : int or tuple of int
        Number of samples in the output (1-D), or a tuple that represents the
        shape of the output (N-D).
    idx : None or int or tuple of int or 'mid', optional
        Index at which the value is 1.  If None, defaults to the 0th element.
        If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in
        all dimensions.  If an int, the impulse will be at `idx` in all
        dimensions.
    dtype : data-type, optional
        The desired data-type for the array, e.g., ``numpy.int8``.  Default is
        ``numpy.float64``.

    Returns
    -------
    y : ndarray
        Output array containing an impulse signal.

    Notes
    -----
    The 1D case is also known as the Kronecker delta.

    Examples
    --------
    An impulse at the 0th element (:math:`\\delta[n]`):

    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> cupyx.scipy.signal.unit_impulse(8)
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    Impulse offset by 2 samples (:math:`\\delta[n-2]`):

    >>> cupyx.scipy.signal.unit_impulse(7, 2)
    array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])

    2-dimensional impulse, centered:

    >>> cupyx.scipy.signal.unit_impulse((3, 3), 'mid')
    array([[ 0.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])

    Impulse at (2, 2), using broadcasting:

    >>> cupyx.scipy.signal.unit_impulse((4, 4), 2)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    out = cupy.empty(shape, dtype)
    shape = np.atleast_1d(shape)

    if idx is None:
        idx = (0,) * len(shape)
    elif idx == 'mid':
        idx = tuple(shape // 2)
    elif not hasattr(idx, "__iter__"):
        idx = (idx,) * len(shape)

    pos = np.ravel_multi_index(idx, out.shape)

    n = out.size
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz

    unit_impulse_kernel = _get_module_func(UNIT_MODULE, 'unit_impulse', out)
    unit_impulse_kernel((n_blocks,), (block_sz,), (n, pos, out))
    return out
