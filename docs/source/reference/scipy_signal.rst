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
   correlation_lags


B-Splines
---------

.. autosummary::
   :toctree: generated/

   gauss_spline
   cspline1d
   qspline1d
   cspline2d
   qspline2d
   cspline1d_eval
   qspline1d_eval
   spline_filter


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
   hilbert
   hilbert2
   decimate
   detrend
   resample
   resample_poly
   upfirdn


Filter design
-------------

.. autosummary::
   :toctree: generated/

   bilinear
   bilinear_zpk
   findfreqs
   freqs
   freqs_zpk
   freqz
   freqz_zpk
   sosfreqz
   firwin
   firwin2
   firls
   minimum_phase
   savgol_coeffs
   gammatone
   group_delay
   iirdesign
   iirfilter
   kaiser_atten
   kaiser_beta
   kaiserord
   unique_roots
   residue
   residuez
   invres
   invresz
   BadCoefficients


Matlab-style IIR filter design
------------------------------

.. autosummary::
   :toctree: generated/

   butter
   buttord
   ellip
   ellipord
   cheby1
   cheb1ord
   cheby2
   cheb2ord
   iircomb
   iirnotch
   iirpeak


Low-level filter design functions
---------------------------------

.. autosummary::
   :toctree: generated/

   abcd_normalize
   band_stop_obj
   buttap
   cheb1ap
   cheb2ap
   ellipap
   lp2bp
   lp2bp_zpk
   lp2bs
   lp2bs_zpk
   lp2hp
   lp2hp_zpk
   lp2lp
   lp2lp_zpk
   normalize


LTI representations
-------------------

.. autosummary::
   :toctree: generated/

   zpk2tf
   zpk2sos
   zpk2ss
   tf2zpk
   tf2sos
   tf2ss
   ss2tf
   ss2zpk
   sos2tf
   sos2zpk
   cont2discrete
   place_poles


Continuous-time linear systems
------------------------------

.. autosummary::
   :toctree: generated/

   lti
   StateSpace
   TransferFunction
   ZerosPolesGain
   lsim
   impulse
   step
   freqresp
   bode


Discrete-time linear systems
----------------------------
.. autosummary::
   :toctree: generated/

   dlti
   StateSpace
   TransferFunction
   ZerosPolesGain
   dlsim
   dimpulse
   dstep
   dfreqresp
   dbode


Waveforms
---------

.. autosummary::
   :toctree: generated/

   chirp
   gausspulse
   max_len_seq
   sawtooth
   square
   unit_impulse


Window functions
----------------
For window functions, see the :mod:`cupyx.scipy.signal.windows` namespace.

In the :mod:`cupyx.scipy.signal` namespace, there is a convenience function
to obtain these windows by name:


.. autosummary::
   :toctree: generated/

   get_window


Wavelets
--------

.. autosummary::
   :toctree: generated/

   morlet
   qmf
   ricker
   morlet2
   cwt


Peak finding
------------

.. autosummary::
   :toctree: generated/

   argrelmin
   argrelmax
   argrelextrema
   find_peaks
   peak_prominences
   peak_widths


Spectral analysis
-----------------

.. autosummary::
   :toctree: generated/

   periodogram
   welch
   csd
   coherence
   spectrogram
   lombscargle
   vectorstrength
   stft
   istft
   check_COLA
   check_NOLA



Chirp Z-transform and Zoom FFT
------------------------------

.. autosummary::
   :toctree: generated/

   czt
   zoom_fft
   CZT
   ZoomFFT
   czt_points
