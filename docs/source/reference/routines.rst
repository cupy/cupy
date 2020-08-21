--------
Routines
--------

The following pages describe NumPy-compatible routines.
These functions cover a subset of
`NumPy routines <https://docs.scipy.org/doc/numpy/reference/routines.html>`_.

.. currentmodule:: cupy

.. toctree::
   :maxdepth: 2

   creation
   manipulation
   binary
   dtype
   fft
   functional
   indexing
   io
   linalg
   logic
   math
   pad
   polynomials
   random
   sorting
   statistics
   ext


CUB/cuTENSOR backend for reduction routines
-------------------------------------------
Some CuPy reduction routines, including :func:`~cupy.sum`, :func:`~cupy.min`, :func:`~cupy.max`,
:func:`~cupy.argmin`, :func:`~cupy.argmax`, and other functions built on top of them, can be
accelerated by switching to the `CUB`_ or `cuTENSOR`_ backend. These backends can be enabled
by setting ``CUPY_ACCELERATORS`` environement variable as documented :ref:`here<environment>`.
Note that while in general the accelerated reductions are faster, there could be exceptions
depending on the data layout. We recommend users to perform some benchmarks to determine
whether CUB/cuTENSOR offers better performance or not.

.. _CUB: https://nvlabs.github.io/cub/
.. _cuTENSOR: https://docs.nvidia.com/cuda/cutensor/index.html
