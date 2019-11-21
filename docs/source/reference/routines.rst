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
   indexing
   io
   linalg
   logic
   math
   pad
   random
   sorting
   statistics
   ext


CUB backend for reduction routines
----------------------------------
Some CuPy reduction routines, including :func:`~cupy.sum`, :func:`~cupy.min`, :func:`~cupy.max`,
:func:`~cupy.argmin`, :func:`~cupy.argmax`, and other functions built on top of them, can be
accelerated by switching to the `CUB backend`_. The switch can be toggled on or off at runtime
by setting the bool :data:`cupy.cuda.cub_enabled`, which is set to ``False`` by default. Note
that while in general CUB-backed reductions are faster, there could be exceptions depending on
the data layout. We recommend users to perform some benchmarks to determine whether CUB offers
better performance or not.

.. _CUB backend: http://nvlabs.github.io/cub/
