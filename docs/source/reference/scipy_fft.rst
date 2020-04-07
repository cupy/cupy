.. module:: cupyx.scipy.fft

Discrete Fourier transforms (``scipy.fft``)
===========================================


Fast Fourier Transforms
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.scipy.fft.fft
   cupyx.scipy.fft.ifft
   cupyx.scipy.fft.fft2
   cupyx.scipy.fft.ifft2
   cupyx.scipy.fft.fftn
   cupyx.scipy.fft.ifftn
   cupyx.scipy.fft.rfft
   cupyx.scipy.fft.irfft
   cupyx.scipy.fft.rfft2
   cupyx.scipy.fft.irfft2
   cupyx.scipy.fft.rfftn
   cupyx.scipy.fft.irfftn
   cupyx.scipy.fft.hfft
   cupyx.scipy.fft.ihfft


Code compatibility features
---------------------------
1. As with other FFT modules in CuPy, FFT functions in this module can take advantage of an existing cuFFT plan (returned by :func:`~cupyx.scipy.fftpack.get_fft_plan`) to accelarate the computation. The plan can be either passed in explicitly via the keyword-only ``plan`` argument or used as a context manager.

2. The boolean switch ``cupy.fft.config.enable_nd_planning`` also affects the FFT functions in this module, see :doc:`./fft`. This switch is neglected when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.

3. Like in ``scipy.fft``, all FFT functions in this module have an optional argument ``overwrite_x`` (default is ``False``), which has the same semantics as in ``scipy.fft``: when it is set to ``True``, the input array ``x`` *can* (not *will*) be overwritten arbitrarily. For this reason, when an in-place FFT is desired, the user should always reassign the input in the following manner: ``x = cupyx.scipy.fftpack.fft(x, ..., overwrite_x=True, ...)``.

4. The ``cupyx.scipy.fft`` module can also be used as a backend for ``scipy.fft`` e.g. by installing with ``scipy.fft.set_backend(cupyx.scipy.fft)``. This can allow ``scipy.fft`` to work with both ``numpy`` and ``cupy`` arrays.

5. The boolean switch :data:`cupy.fft.config.use_multi_gpus` also affects the FFT functions in this module, see :doc:`./fft`. Moreover, this switch is *honored* when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.

.. note::
   ``scipy.fft`` requires SciPy version 1.4.0 or newer.

.. note::
    To use ``scipy.fft.set_backend()`` together with an explicit ``plan`` argument requires SciPy version 1.5.0 or newer.
