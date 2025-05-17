:orphan:

.. This page is to generate documentation for deprecated APIs removed from the
   public table of contents.

NumPy Routines
--------------

.. autosummary::
   :toctree: generated/

   # Removed in NumPy v2.0
   cupy.asfarray

   # Marked deprecated in NumPy v2.0
   cupy.in1d
   cupy.row_stack
   cupy.trapz

DLPack helper
-------------

.. autosummary::
   :toctree: generated/

   cupy.fromDlpack

Time range
----------

.. autosummary::
   :toctree: generated/

   cupy.prof.TimeRangeDecorator
   cupy.prof.time_range

Timing helper
-------------

.. autosummary::
   :toctree: generated/

   cupyx.time.repeat

Profiler
--------

.. autosummary::
   :toctree: generated/

   cupy.cuda.profile
   cupy.cuda.profiler.start
   cupy.cuda.profiler.stop

Device synchronization detection
--------------------------------

.. warning::

   These APIs are deprecated in CuPy v10 and will be removed in future releases.

.. autosummary::
   :toctree: generated/

   cupyx.allow_synchronize
   cupyx.DeviceSynchronized
