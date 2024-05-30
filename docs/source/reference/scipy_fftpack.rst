.. module:: cupyx.scipy.fftpack

Legacy discrete fourier transforms (:mod:`cupyx.scipy.fftpack`)
===============================================================

.. note::

   As of SciPy version 1.4.0, :mod:`scipy.fft` is recommended over
   :mod:`scipy.fftpack`. Consider using :mod:`cupyx.scipy.fft` instead.

.. Hint:: `SciPy API Reference: Legacy discrete Fourier transforms (scipy.fftpack) <https://docs.scipy.org/doc/scipy/reference/fftpack.html>`_

Fast Fourier Transforms (FFTs)
------------------------------

.. autosummary::
   :toctree: generated/

   fft
   ifft
   fft2
   ifft2
   fftn
   ifftn
   rfft
   irfft
   get_fft_plan


Code compatibility features
---------------------------
1. As with other FFT modules in CuPy, FFT functions in this module can take advantage of an existing cuFFT plan (returned by :func:`~cupyx.scipy.fftpack.get_fft_plan`) to accelerate the computation. The plan can be either passed in explicitly via the ``plan`` argument or used as a context manager. The argument ``plan`` is currently experimental and the interface may be changed in the future version. The :func:`~cupyx.scipy.fftpack.get_fft_plan` function has no counterpart in ``scipy.fftpack``.

2. The boolean switch :data:`cupy.fft.config.enable_nd_planning` also affects the FFT functions in this module, see :doc:`./fft`. This switch is neglected when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.

3. Like in ``scipy.fftpack``, all FFT functions in this module have an optional argument ``overwrite_x`` (default is ``False``), which has the same semantics as in ``scipy.fftpack``: when it is set to ``True``, the input array ``x`` *can* (not *will*) be overwritten arbitrarily. For this reason, when an in-place FFT is desired, the user should always reassign the input in the following manner: ``x = cupyx.scipy.fftpack.fft(x, ..., overwrite_x=True, ...)``.

4. The boolean switch :data:`cupy.fft.config.use_multi_gpus` also affects the FFT functions in this module, see :doc:`./fft`. Moreover, this switch is *honored* when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.
