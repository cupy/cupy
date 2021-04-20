CuPy-specific Functions
=======================

CuPy-specific functions are placed under ``cupyx`` namespace.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.rsqrt
   cupyx.scatter_add
   cupyx.scatter_max
   cupyx.scatter_min
   cupyx.empty_pinned
   cupyx.empty_like_pinned
   cupyx.zeros_pinned
   cupyx.zeros_like_pinned

DLPack utilities
----------------

This is a helper function for creating a :class:`cupy.ndarray` from a DLPack tensor.
For further detail see :ref:`dlpack`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fromDlpack


.. _kernel_param_opt:

Automatic Kernel Parameters Optimizations
-----------------------------------------

.. module:: cupyx.optimizing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupyx.optimizing.optimize
