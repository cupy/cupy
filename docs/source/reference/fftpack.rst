.. module:: cupyx.scipy.fftpack

Discrete Fourier transforms
===========================


Fast Fourier Transforms
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.fftpack.fft
   cupyx.scipy.fftpack.ifft
   cupyx.scipy.fftpack.fft2
   cupyx.scipy.fftpack.ifft2
   cupyx.scipy.fftpack.fftn
   cupyx.scipy.fftpack.ifftn
   cupyx.scipy.fftpack.rfft
   cupyx.scipy.fftpack.irfft
   cupyx.scipy.fftpack.get_fft_plan


Code compatibility features
---------------------------
The ``get_fft_plan`` function has no counterpart in ``scipy.fftpack``. It returns a cuFFT plan that can be passed to the FFT functions in this module (using the argument ``plan``) to accelarate the computation.

Like in ``scipy.fftpack``, all FFT functions in this module have an optional argument ``overwrite_x`` (default is ``False``), which has the same semantics as in ``scipy.fftpack``: when it is set to ``True``, the input array ``x`` *can* (not *will*) be destroyed and replaced by the output. Therefore, to guarantee an in-place FFT is successfully performed, one should always re-assign the input: ``x = cupyx.scipy.fftpack.fft(x, ..., overwrite_x=True, ...)``.
