Memory Management
=================

CuPy uses *memory pool* for memory allocations by default.
The memory pool significantly improves the performance by mitigating the overhead of memory allocation and CPU/GPU synchronization.

There are two different memory pools in CuPy:

* Device memory pool (GPU device memory), which is used for GPU memory allocations.
* Pinned memory pool (non-swappable CPU memory), which is used during CPU-to-GPU data transfer.

.. attention::

   When you monitor the memory usage (e.g., using ``nvidia-smi`` for GPU memory or ``ps`` for CPU memory), you may notice that memory not being freed even after the array instance become out of scope.
   This is an expected behavior, as the default memory pool "caches" the allocated memory blocks.

See :doc:`cuda` for the details of memory management APIs.

Memory Pool Operations
----------------------

The memory pool instance provides statistics about memory allocation.
To access the default memory pool instance, use :func:`cupy.get_default_memory_pool` and :func:`cupy.get_default_pinned_memory_pool`.
You can also free all unused memory blocks hold in the memory pool.
See the example code below for details:

.. code-block:: py

   import numpy
   import cupy

   mempool = cupy.get_default_memory_pool()
   pinned_mempool = cupy.get_default_pinned_memory_pool()

   # Create an array on CPU.
   # NumPy allocates 400 bytes in CPU (not managed by CuPy memory pool).
   a_cpu = numpy.ndarray(100, dtype=numpy.float32)

   # You can access statistics of these memory pools.
   print(mempool.used_bytes())              # 0
   print(mempool.total_bytes())             # 0
   print(pinned_mempool.n_free_blocks())    # 0

   # Transfer the array from CPU to GPU.
   # This allocates 400 bytes on from the device memory pool, and another 400
   # bytes from the pinned memory pool.  The allocated pinned memory will be
   # released just after the transfer is complete.  Note that the actual
   # allocation size may be rounded to larger value than the requested size
   # for performance.
   a = cupy.array(a_cpu)
   print(mempool.used_bytes())              # 512
   print(mempool.total_bytes())             # 512
   print(pinned_mempool.n_free_blocks())    # 1

   # When the array goes out of scope, the allocated device memory is released
   # and kept in the pool for future reuse.
   a = None  # (or `del a`)
   print(mempool.used_bytes())              # 0
   print(mempool.total_bytes())             # 512
   print(pinned_mempool.n_free_blocks())    # 1

   # You can clear the memory pool by calling `free_all_blocks`.
   mempool.free_all_blocks()
   pinned_mempool.free_all_blocks()
   print(mempool.used_bytes())              # 0
   print(mempool.total_bytes())             # 0
   print(pinned_mempool.n_free_blocks())    # 0

Changing Memory Pool
--------------------

You can use your own memory allocator instead of the default memory pool by passing the memory allocation function to :func:`cupy.cuda.set_allocator` / :func:`cupy.cuda.set_pinned_memory_allocator`.
The memory allocator function should take 1 argument (the requested size in bytes) and return :class:`cupy.cuda.MemoryPointer` / :class:`cupy.cuda.PinnedMemoryPointer`.

You can even disable the default memory pool by the code below.
Be sure to do this before any other CuPy operations.

.. code-block:: py

   import cupy

   # Disable memory pool for device memory (GPU)
   cupy.cuda.set_allocator(None)

   # Disable memory pool for pinned memory (CPU).
   cupy.cuda.set_pinned_memory_allocator(None)
