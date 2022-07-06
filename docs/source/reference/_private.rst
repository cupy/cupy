:orphan:

.. This page is to generate documentation for private classes exposed to users,
   i.e., users cannot instantiate it by themselves but may use it's properties
   or methods via returned values from CuPy methods.
   These classes must be referred in public APIs returning their instances.

Benchmark Data
--------------

.. autosummary::
   :toctree: generated/

   cupyx.profiler._time._PerfCaseResult

cuFFT Plan Cache
----------------

.. autosummary::
   :toctree: generated/

   cupy.fft._cache.PlanCache

JIT Cooperative Groups
----------------------

.. autosummary::
   :toctree: generated/

   cupyx.jit.cg._ThreadBlockGroup
   cupyx.jit.cg._GridGroup
