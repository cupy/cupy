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
1. The ``get_fft_plan`` function has no counterpart in ``scipy.fftpack``. It returns a cuFFT plan that can be passed to the FFT functions in this module (using the argument ``plan``) to accelarate the computation. The argument ``plan`` is currently experimental and the interface may be changed in the future version.

2. The boolean switch ``cupy.fft.config.enable_nd_planning`` also affects the FFT functions in this module, see :doc:`./fft`. This switch is neglected when planning manually using ``get_fft_plan``.

3. Like in ``scipy.fftpack``, all FFT functions in this module have an optional argument ``overwrite_x`` (default is ``False``), which has the same semantics as in ``scipy.fftpack``: when it is set to ``True``, the input array ``x`` *can* (not *will*) be destroyed and replaced by the output. For this reason, when an in-place FFT is desired, the user should always reassign the input in the following manner: ``x = cupyx.scipy.fftpack.fft(x, ..., overwrite_x=True, ...)``.
