----------------
Routines (NumPy)
----------------

The following pages describe NumPy-compatible routines.
These functions cover a subset of
`NumPy routines <https://numpy.org/doc/stable/reference/routines.html>`_.

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
   misc
   pad
   polynomials
   random
   set
   sorting
   statistics
   testing
   window


.. _accelerators:

CUB/cuTENSOR backend for some CuPy routines
-------------------------------------------
Some CuPy reduction routines, including :func:`~cupy.sum`, :func:`~cupy.amin`, :func:`~cupy.amax`,
:func:`~cupy.argmin`, :func:`~cupy.argmax`, and other functions built on top of them, can be
accelerated by switching to the `CUB`_ or `cuTENSOR`_ backend. These backends can be enabled
by setting the ``CUPY_ACCELERATORS`` environement variable as documented :ref:`here<environment>`.
Note that while in general the accelerated reductions are faster, there could be exceptions
depending on the data layout. In particular, the CUB reduction only supports reduction over
contiguous axes.

CUB also accelerates other routines, such as inclusive scans (ex: :func:`~cupy.cumsum`), histograms,
sparse matrix-vector multiplications (not applicable in CUDA 11), and :class:`cupy.ReductionKernel`.

In any case, we recommend users to perform some benchmarks to determine whether CUB/cuTENSOR offers
better performance or not.

.. _CUB: https://nvlabs.github.io/cub/
.. _cuTENSOR: https://docs.nvidia.com/cuda/cutensor/index.html
