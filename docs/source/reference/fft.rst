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
   cupy.fft.config.set_cufft_gpus


Normalization
-------------
The default normalization has the direct transforms unscaled and the inverse transforms are scaled by :math:`1/n`.
If the ketyword argument ``norm`` is ``"ortho"``, both transforms will be scaled by :math:`1/\sqrt{n}`.


Code compatibility features
---------------------------
FFT functions of NumPy alway return numpy.ndarray which type is ``numpy.complex128`` or ``numpy.float64``.
CuPy functions do not follow the behavior, they will return ``numpy.complex64`` or ``numpy.float32`` if the type of the input is ``numpy.float16``, ``numpy.float32``, or ``numpy.complex64``.

Internally, ``cupy.fft`` always generates a *cuFFT plan* (see the `cuFFT documentation`_ for detail) corresponding to the desired transform. When possible, an n-dimensional plan will be used, as opposed to applying separate 1D plans for each axis to be transformed. Using n-dimensional planning can provide better performance for multidimensional transforms, but requires more GPU memory than separable 1D planning. The user can disable n-dimensional planning by setting ``cupy.fft.config.enable_nd_planning = False``. This ability to adjust the planning type is a deviation from the NumPy API, which does not use precomputed FFT plans.

Moreover, the automatic plan generation can be suppressed by using an existing plan returned by :func:`cupyx.scipy.fftpack.get_fft_plan` as a context manager. This is again a deviation from NumPy.


Multi-GPU FFT
-------------
:mod:`cupy.fft` can use multiple GPUs. To enable (disable) this feature, set :data:`cupy.fft.config.use_multi_gpus` to ``True`` (``False``). Next, to set the number of GPUs or the participating GPU IDs, use the function :func:`cupy.fft.config.set_cufft_gpus`. All of the limitations listed in the `cuFFT documentation`_ apply here. In particular, using more than one GPU does not guarantee better performance.


.. _cuFFT documentation: https://docs.nvidia.com/cuda/cufft/index.html
