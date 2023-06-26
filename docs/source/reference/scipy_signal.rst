.. module:: cupyx.scipy.signal

Signal processing (:mod:`cupyx.scipy.signal`)
=============================================

.. Hint:: `SciPy API Reference: Signal processing (scipy.signal) <https://docs.scipy.org/doc/scipy/reference/signal.html>`_

Convolution
-----------

.. autosummary::
   :toctree: generated/

   convolve
   correlate
   fftconvolve
   oaconvolve
   convolve2d
   correlate2d
   sepfir2d
   choose_conv_method


Filtering
---------

.. autosummary::
   :toctree: generated/

   order_filter
   medfilt
   medfilt2d
   wiener
   symiirorder1
   symiirorder2
   lfilter
   lfiltic
   lfilter_zi
   filtfilt
   savgol_filter
   deconvolve
   sosfilt
   sosfilt_zi
   sosfiltfilt
   detrend
   hilbert
   hilbert2


Filter design
-------------

.. autosummary::
   :toctree: generated/

   bilinear
   bilinear_zpk
   freqz
   freqz_zpk
   firls
   minimum_phase
   savgol_coeffs
   iirfilter


Matlab-style IIR filter design
------------------------------

.. autosummary::
   :toctree: generated/

   butter
   ellip
   cheby1
   cheby2


Low-level filter design functions
---------------------------------

.. autosummary::
   :toctree: generated/

   abcd_normalize



Chirp Z-transform and Zoom FFT
------------------------------

.. autosummary::
   :toctree: generated/

   czt
   zoom_fft
   CZT
   ZoomFFT
   czt_points


LTI representations
-------------------

.. autosummary::
   :toctree: generated/

   zpk2tf
   zpk2sos

