# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pytest
import numpy

import cupy
from cupy import testing
import cupyx.signal


# https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
def complex_cepstrum(x, n=None):
    """Compute the complex cepstrum of a real sequence.
    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.
    The complex cepstrum is given by
    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}
    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    --------
    """

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = numpy.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = numpy.array(numpy.round(unwrapped[..., center] / numpy.pi))
        unwrapped -= (numpy.pi * ndelay[..., None] * numpy.arange(samples)
                      / center)
        return unwrapped, ndelay

    spectrum = numpy.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(numpy.angle(spectrum))
    log_spectrum = numpy.log(numpy.abs(spectrum)) + 1j * unwrapped_phase
    ceps = numpy.fft.ifft(log_spectrum).real

    return ceps, ndelay


def real_cepstrum(x, n=None):
    """
    Compute the real cepstrum of a real sequence.
    x : ndarray
        Real sequence to compute real cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps: ndarray
        The real cepstrum.
    """
    spectrum = numpy.fft.fft(x, n=n)
    ceps = numpy.fft.ifft(numpy.log(numpy.abs(spectrum))).real

    return ceps


def inverse_complex_cepstrum(ceps, ndelay):
    r"""Compute the inverse complex cepstrum of a real sequence.
    ceps : ndarray
        Real sequence to compute inverse complex cepstrum of.
    ndelay: int
        The amount of samples of circular delay added to `x`.
    Returns
    -------
    x : ndarray
        The inverse complex cepstrum of the real sequence `ceps`.
    The inverse complex cepstrum is given by
    .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}
    where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    """

    def _wrap(phase, ndelay):
        ndelay = numpy.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = (
            phase + numpy.pi * ndelay[..., None] * numpy.arange(samples)
            / center
        )
        return wrapped

    log_spectrum = numpy.fft.fft(ceps)
    spectrum = numpy.exp(
        log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay)
    )
    x = numpy.fft.ifft(spectrum).real

    return x


def minimum_phase(x, n=None):
    r"""Compute the minimum phase reconstruction of a real sequence.
    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.
    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.
    """
    if n is None:
        n = len(x)
    ceps = real_cepstrum(x, n=n)
    odd = n % 2
    window = numpy.concatenate(
        (
            [1.0],
            2.0 * numpy.ones((n + odd) // 2 - 1),
            numpy.ones(1 - odd),
            numpy.zeros((n + odd) // 2 - 1),
        )
    )

    m = numpy.fft.ifft(numpy.exp(numpy.fft.fft(window * ceps))).real

    return m


@pytest.mark.parametrize("num_samps", [2**8, 2**14])
@pytest.mark.parametrize("n", [123, 256])
def test_complex_cepstrum(num_samps, n):
    cpu_sig = numpy.random.rand(num_samps)
    gpu_sig = cupy.array(cpu_sig)
    gpu_out = cupyx.signal.complex_cepstrum(gpu_sig, n)
    cpu_out = complex_cepstrum(cpu_sig, n)
    testing.assert_allclose(cpu_out[0], gpu_out[0])
    testing.assert_allclose(cpu_out[1], gpu_out[1])


@pytest.mark.parametrize("num_samps", [2**8, 2**14])
@pytest.mark.parametrize("n", [123, 256])
def test_real_cepstrum(num_samps, n):
    cpu_sig = numpy.random.rand(num_samps)
    gpu_sig = cupy.array(cpu_sig)
    gpu_out = cupyx.signal.real_cepstrum(gpu_sig, n)
    cpu_out = real_cepstrum(cpu_sig, n)
    testing.assert_allclose(cpu_out, gpu_out)


@pytest.mark.parametrize("num_samps", [2**10])
@pytest.mark.parametrize("n", [123, 256])
def test_inverse_complex_cepstrum(num_samps, n):
    cpu_sig = numpy.random.rand(num_samps)
    gpu_sig = cupy.array(cpu_sig)
    gpu_out = cupyx.signal.inverse_complex_cepstrum(gpu_sig, n)
    cpu_out = inverse_complex_cepstrum(cpu_sig, n)
    testing.assert_allclose(cpu_out, gpu_out)


@pytest.mark.parametrize("num_samps", [2**8, 2**14])
@pytest.mark.parametrize("n", [123, 256])
def test_minimum_phase(num_samps, n):
    cpu_sig = numpy.random.rand(num_samps)
    gpu_sig = cupy.array(cpu_sig)
    gpu_out = cupyx.signal.minimum_phase(gpu_sig, n)
    cpu_out = minimum_phase(cpu_sig, n)
    testing.assert_allclose(cpu_out, gpu_out)
