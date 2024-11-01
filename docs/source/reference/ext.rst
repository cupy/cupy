CuPy-specific functions
=======================

CuPy-specific functions are placed under ``cupyx`` namespace.

.. TODO(kmaehashi): use module:: cupyx
.. autosummary::
   :toctree: generated/

   cupyx.rsqrt
   cupyx.scatter_add
   cupyx.scatter_max
   cupyx.scatter_min
   cupyx.empty_pinned
   cupyx.empty_like_pinned
   cupyx.zeros_pinned
   cupyx.zeros_like_pinned

non-SciPy compat Signal API
---------------------------

The functions under `cupyx.signal` are non-SciPy compat signal API ported from cuSignal 
through the courtesy of Nvidia cuSignal team.

.. autosummary::
   :toctree: generated/

   cupyx.signal.channelize_poly
   cupyx.signal.convolve1d3o
   cupyx.signal.pulse_compression
   cupyx.signal.pulse_doppler
   cupyx.signal.cfar_alpha
   cupyx.signal.ca_cfar
   cupyx.signal.freq_shift
   
Profiling utilities
-------------------

.. autosummary::
   :toctree: generated/

   cupyx.profiler.benchmark
   cupyx.profiler.time_range
   cupyx.profiler.profile

DLPack utilities
----------------

Below are helper functions for creating a :class:`cupy.ndarray` from either a DLPack tensor
or any object supporting the DLPack data exchange protocol.
For further detail see :ref:`dlpack`.

.. autosummary::
   :toctree: generated/

   cupy.from_dlpack


.. _kernel_param_opt:

Automatic Kernel Parameters Optimizations (:mod:`cupyx.optimizing`)
-------------------------------------------------------------------

.. module:: cupyx.optimizing
.. autosummary::
   :toctree: generated/

   cupyx.optimizing.optimize
