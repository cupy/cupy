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

DLPack utilities
----------------

This is a helper function for creating a :class:`cupy.ndarray` from a DLPack tensor.
For further detail see :ref:`dlpack`.

.. autosummary::
   :toctree: generated/

   cupy.fromDlpack


.. _kernel_param_opt:

Automatic Kernel Parameters Optimizations (:mod:`cupyx.optimizing`)
-------------------------------------------------------------------

.. module:: cupyx.optimizing
.. autosummary::
   :toctree: generated/

   cupyx.optimizing.optimize
