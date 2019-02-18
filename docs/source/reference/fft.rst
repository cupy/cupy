.. module:: cupy.fft

FFT Functions
=============

.. https://docs.scipy.org/doc/numpy/reference/routines.fft.html

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fft.fft
   cupy.fft.ifft
   cupy.fft.fft2
   cupy.fft.ifft2
   cupy.fft.fftn
   cupy.fft.ifftn


Real FFTs
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fft.rfft
   cupy.fft.irfft
   cupy.fft.rfft2
   cupy.fft.irfft2
   cupy.fft.rfftn
   cupy.fft.irfftn


Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fft.hfft
   cupy.fft.ihfft


Helper routines
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fft.fftfreq
   cupy.fft.rfftfreq
   cupy.fft.fftshift
   cupy.fft.ifftshift


Normalization
-------------
The default normalization has the direct transforms unscaled and the inverse transforms are scaled by :math:`1/n`.
If the ketyword argument ``norm`` is ``"ortho"``, both transforms will be scaled by :math:`1/\sqrt{n}`.


Code compatibility features
---------------------------
FFT functions of NumPy alway return numpy.ndarray which type is ``numpy.complex128`` or ``numpy.float64``.
CuPy functions do not follow the behavior, they will return ``numpy.complex64`` or ``numpy.float32`` if the type of the input is ``numpy.float16``, ``numpy.float32``, or ``numpy.complex64``.

In addition, when transforming over more than 1 axis ``cupy.fft`` will attempt to generate a *cuFFT plan* internally (see the `cuFFT documentation`_ for detail) to accelarate the computation. This is enabled by default but can be turned off by setting ``cupy.fft.config.enable_nd_planning = False``. This feature is a deviation from NumPy which has no planning.

.. _cuFFT documentation: https://docs.nvidia.com/cuda/cufft/index.html
