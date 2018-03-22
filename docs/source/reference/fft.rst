FFT Functions
=============


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
