.. module:: cupyx.scipy.fft

Discrete Fourier transforms (:mod:`cupyx.scipy.fft`)
====================================================

.. Hint:: `SciPy API Reference: Discrete Fourier transforms (scipy.fft) <https://docs.scipy.org/doc/scipy/reference/fft.html>`_

.. seealso:: :doc:`../user_guide/fft`

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
   rfft2
   irfft2
   rfftn
   irfftn
   hfft
   ihfft
   hfft2
   ihfft2
   hfftn
   ihfftn

Discrete Cosine and Sine Transforms (DST and DCT)
-------------------------------------------------

.. autosummary::
   :toctree: generated/

   dct
   idct
   dctn
   idctn
   dst
   idst
   dstn
   idstn

Fast Hankel Transforms
----------------------

.. autosummary::
   :toctree: generated/

   fht
   ifht

Helper functions
----------------

.. autosummary::
   :toctree: generated/

   fftshift
   ifftshift
   fftfreq
   rfftfreq
   next_fast_len


Code compatibility features
---------------------------
1. As with other FFT modules in CuPy, FFT functions in this module can take advantage of an existing cuFFT plan (returned by :func:`~cupyx.scipy.fftpack.get_fft_plan`) to accelerate the computation. The plan can be either passed in explicitly via the keyword-only ``plan`` argument or used as a context manager. One exception to this are the DCT and DST transforms, which do not currently support a plan argument.

2. The boolean switch ``cupy.fft.config.enable_nd_planning`` also affects the FFT functions in this module, see :doc:`./fft`. This switch is neglected when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.

3. Like in ``scipy.fft``, all FFT functions in this module have an optional argument ``overwrite_x`` (default is ``False``), which has the same semantics as in ``scipy.fft``: when it is set to ``True``, the input array ``x`` *can* (not *will*) be overwritten arbitrarily. For this reason, when an in-place FFT is desired, the user should always reassign the input in the following manner: ``x = cupyx.scipy.fftpack.fft(x, ..., overwrite_x=True, ...)``.

4. The ``cupyx.scipy.fft`` module can also be used as a backend for ``scipy.fft`` e.g. by installing with ``scipy.fft.set_backend(cupyx.scipy.fft)``. This can allow ``scipy.fft`` to work with both ``numpy`` and ``cupy`` arrays. For more information, see :ref:`scipy_fft_backend`.

5. The boolean switch :data:`cupy.fft.config.use_multi_gpus` also affects the FFT functions in this module, see :doc:`./fft`. Moreover, this switch is *honored* when planning manually using :func:`~cupyx.scipy.fftpack.get_fft_plan`.

6. Both type II and III DCT and DST transforms are implemented. Type I and IV transforms are currently unavailable.
