.. currentmodule:: cupy

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.

CuPy v9
=======

Dropping Support of CUDA 9.0
----------------------------

CUDA 9.0 is no longer supported.
Use CUDA 9.2 or later.

Dropping Support of cuDNN v7.5 and NCCL v2.3
--------------------------------------------

cuDNN v7.5 (or earlier) and NCCL v2.3 (or earlier) are no longer supported.

Dropping Support of NumPy 1.16 and SciPy 1.3
--------------------------------------------

NumPy 1.16 and SciPy 1.3 are no longer supported.

Dropping Support of Python 3.5
------------------------------

Python 3.5 is no longer supported in CuPy v9.

NCCL and cuDNN No Longer Included in Wheels
-------------------------------------------

NCCL and cuDNN shared libraires are no longer included in wheels (see `#4850 <https://github.com/cupy/cupy/issues/4850>`_ for discussions). 
You can manually install them after installing wheel if you don't have a previous installation; see :doc:`install` for details.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 11.2 and Python 3.8.


CuPy v8
=======

Dropping Support of CUDA 8.0 and 9.1
------------------------------------

CUDA 8.0 and 9.1 are no longer supported.
Use CUDA 9.0, 9.2, 10.0, or later.

Dropping Support of NumPy 1.15 and SciPy 1.2
--------------------------------------------

NumPy 1.15 (or earlier) and SciPy 1.2 (or earlier) are no longer supported.

Update of Docker Images
-----------------------

* CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 10.2 and Python 3.6.
* SciPy and Optuna are now pre-installed.

CUB Support and Compiler Requirement
------------------------------------

CUB module is now built by default.
You can enable the use of CUB by setting ``CUPY_ACCELERATORS="cub"`` (see :doc:`reference/environment` for details).

Due to this change, g++-6 or later is required when building CuPy from the source.
See :doc:`install` for details.

The following environment variables are no longer effective:

* ``CUB_DISABLED``: Use ``CUPY_ACCELERATORS`` as aforementioned.
* ``CUB_PATH``: No longer required as CuPy uses either the CUB source bundled with CUDA (only when using CUDA 11.0 or later) or the one in the CuPy distribution.

API Changes
-----------

* ``cupy.scatter_add``, which was deprecated in CuPy v4, has been removed. Use :func:`cupyx.scatter_add` instead.
* ``cupy.sparse`` module has been deprecated and will be removed in future releases. Use :mod:`cupyx.scipy.sparse` instead.
* ``dtype`` argument of :func:`cupy.ndarray.min` and :func:`cupy.ndarray.max` has been removed to align with the NumPy specification.
* :func:`cupy.allclose` now returns the result as 0-dim GPU array instead of Python bool to avoid device synchronization.
* :class:`cupy.RawModule` now delays the compilation to the time of the first call to align the behavior with :class:`cupy.RawKernel`.
* ``cupy.cuda.*_enabled`` flags (``nccl_enabled``, ``nvtx_enabled``, etc.) has been deprecated. Use ``cupy.cuda.*.available`` flag (``cupy.cuda.nccl.available``, ``cupy.cuda.nvtx.available``, etc.) instead.
* ``CHAINER_SEED`` environment variable is no longer effective. Use ``CUPY_SEED`` instead.


CuPy v7
=======

Dropping Support of Python 2.7 and 3.4
--------------------------------------

Starting from CuPy v7, Python 2.7 and 3.4 are no longer supported as it reaches its end-of-life (EOL) in January 2020 (2.7) and March 2019 (3.4).
Python 3.5.1 is the minimum Python version supported by CuPy v7.
Please upgrade the Python version if you are using affected versions of Python to any later versions listed under :doc:`install`.


CuPy v6
=======

Binary Packages Ignore ``LD_LIBRARY_PATH``
------------------------------------------

Prior to CuPy v6, ``LD_LIBRARY_PATH`` environment variable can be used to override cuDNN / NCCL libraries bundled in the binary distribution (also known as wheels).
In CuPy v6, ``LD_LIBRARY_PATH`` will be ignored during discovery of cuDNN / NCCL; CuPy binary distributions always use libraries that comes with the package to avoid errors caused by unexpected override.


CuPy v5
=======

``cupyx.scipy`` Namespace
-------------------------

:mod:`cupyx.scipy` namespace has been introduced to provide CUDA-enabled SciPy functions.
:mod:`cupy.sparse` module has been renamed to :mod:`cupyx.scipy.sparse`; :mod:`cupy.sparse` will be kept as an alias for backward compatibility.

Dropped Support for CUDA 7.0 / 7.5
----------------------------------

CuPy v5 no longer supports CUDA 7.0 / 7.5.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 9.2 and cuDNN 7.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.


CuPy v4
=======

.. note::

   The version number has been bumped from v2 to v4 to align with the versioning of Chainer.
   Therefore, CuPy v3 does not exist.

Default Memory Pool
-------------------

Prior to CuPy v4, memory pool was only enabled by default when CuPy is used with Chainer.
In CuPy v4, memory pool is now enabled by default, even when you use CuPy without Chainer.
The memory pool significantly improves the performance by mitigating the overhead of memory allocation and CPU/GPU synchronization.

.. attention::

   When you monitor GPU memory usage (e.g., using ``nvidia-smi``), you may notice that GPU memory not being freed even after the array instance become out of scope.
   This is expected behavior, as the default memory pool "caches" the allocated memory blocks.

To access the default memory pool instance, use :func:`get_default_memory_pool` and :func:`get_default_pinned_memory_pool`.
You can access the statistics and free all unused memory blocks "cached" in the memory pool.

.. code-block:: py

   import cupy
   a = cupy.ndarray(100, dtype=cupy.float32)
   mempool = cupy.get_default_memory_pool()

   # For performance, the size of actual allocation may become larger than the requested array size.
   print(mempool.used_bytes())   # 512
   print(mempool.total_bytes())  # 512

   # Even if the array goes out of scope, its memory block is kept in the pool.
   a = None
   print(mempool.used_bytes())   # 0
   print(mempool.total_bytes())  # 512

   # You can clear the memory block by calling `free_all_blocks`.
   mempool.free_all_blocks()
   print(mempool.used_bytes())   # 0
   print(mempool.total_bytes())  # 0

You can even disable the default memory pool by the code below.
Be sure to do this before any other CuPy operations.

.. code-block:: py

   import cupy
   cupy.cuda.set_allocator(None)
   cupy.cuda.set_pinned_memory_allocator(None)

Compute Capability
------------------

CuPy v4 now requires NVIDIA GPU with Compute Capability 3.0 or larger.
See the `List of CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_ to check if your GPU supports Compute Capability 3.0.


CUDA Stream
-----------

As CUDA Stream is fully supported in CuPy v4, ``cupy.cuda.RandomState.set_stream``, the function to change the stream used by the random number generator, has been removed.
Please use :func:`cupy.cuda.Stream.use` instead.

See the discussion in `#306 <https://github.com/cupy/cupy/pull/306>`_ for more details.

``cupyx`` Namespace
-------------------

``cupyx`` namespace has been introduced to provide features specific to CuPy (i.e., features not provided in NumPy) while avoiding collision in future.
See :doc:`reference/ext` for the list of such functions.

For this rule, :func:`cupy.scatter_add` has been moved to :func:`cupyx.scatter_add`.
:func:`cupy.scatter_add` is still available as an alias, but it is encouraged to use :func:`cupyx.scatter_add` instead.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 8.0 and cuDNN 6.0.
This change was introduced because CUDA 7.5 does not support NVIDIA Pascal GPUs.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.

CuPy v2
=======

Changed Behavior of count_nonzero Function
------------------------------------------

For performance reasons, :func:`cupy.count_nonzero` has been changed to return zero-dimensional :class:`ndarray` instead of `int` when `axis=None`.
See the discussion in `#154 <https://github.com/cupy/cupy/pull/154>`_ for more details.
