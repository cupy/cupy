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
