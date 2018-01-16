.. currentmodule:: cupy

=============
Upgrade Guide
=============

This is a list of changes introduced in each release that users should be aware of when migrating from older versions.
Most changes are carefully designed not to break existing code; however changes that may possibly break them are highlighted with a box.


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
   cupy.cuda.set_allocator()
   cupy.cuda.set_pinned_memory_allocator()

Compute Capability
------------------

CuPy v4 now requires NVIDIA GPU with Compute Capability 3.0 or larger.
See the `List of CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_ to check if your GPU supports Compute Capability 3.0.


CUDA Stream
-----------

As CUDA Stream is fully supported in CuPy v4, ``cupy.cuda.RandomState.set_stream``, the function to change the stream used by the random number generator, has been removed.
Please use :func:`cupy.cuda.Stream.use` instead.

See the discussion in `#306 <https://github.com/cupy/cupy/pull/306>`_ for more details.


CuPy v2
=======

Changed Behavior of count_nonzero Function
------------------------------------------

For performance reasons, :func:`cupy.count_nonzero` has been changed to return zero-dimensional :class:`ndarray` instead of `int` when `axis=None`.
See the discussion in `#154 <https://github.com/cupy/cupy/pull/154>`_ for more details.
