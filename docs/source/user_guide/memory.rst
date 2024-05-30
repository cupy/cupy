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

See :doc:`../reference/cuda` for the details of memory management APIs.

For using pinned memory more conveniently, we also provide a few high-level APIs in the ``cupyx`` namespace,
including :func:`cupyx.empty_pinned`, :func:`cupyx.empty_like_pinned`, :func:`cupyx.zeros_pinned`, and
:func:`cupyx.zeros_like_pinned`. They return NumPy arrays backed by pinned memory. If CuPy's pinned memory pool
is in use, the pinned memory is allocated from the pool.

.. note::

    CuPy v8 and above provides a :ref:`FFT plan cache <fft_plan_cache>` that could use a portion of device memory if FFT and related functions are used.
    The memory taken can be released by shrinking or disabling the cache.


Memory Pool Operations
----------------------

The memory pool instance provides statistics about memory allocation.
To access the default memory pool instance, use :func:`cupy.get_default_memory_pool` and :func:`cupy.get_default_pinned_memory_pool`.
You can also free all unused memory blocks hold in the memory pool.
See the example code below for details:

.. code-block:: py

   import cupy
   import numpy

   mempool = cupy.get_default_memory_pool()
   pinned_mempool = cupy.get_default_pinned_memory_pool()

   # Create an array on CPU.
   # NumPy allocates 400 bytes in CPU (not managed by CuPy memory pool).
   a_cpu = numpy.ndarray(100, dtype=numpy.float32)
   print(a_cpu.nbytes)                      # 400

   # You can access statistics of these memory pools.
   print(mempool.used_bytes())              # 0
   print(mempool.total_bytes())             # 0
   print(pinned_mempool.n_free_blocks())    # 0

   # Transfer the array from CPU to GPU.
   # This allocates 400 bytes from the device memory pool, and another 400
   # bytes from the pinned memory pool.  The allocated pinned memory will be
   # released just after the transfer is complete.  Note that the actual
   # allocation size may be rounded to larger value than the requested size
   # for performance.
   a = cupy.array(a_cpu)
   print(a.nbytes)                          # 400
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

See :class:`cupy.cuda.MemoryPool` and :class:`cupy.cuda.PinnedMemoryPool` for details.

Limiting GPU Memory Usage
-------------------------

You can hard-limit the amount of GPU memory that can be allocated by using ``CUPY_GPU_MEMORY_LIMIT`` environment variable (see :doc:`../reference/environment` for details).

.. code-block:: py

   # Set the hard-limit to 1 GiB:
   #   $ export CUPY_GPU_MEMORY_LIMIT="1073741824"

   # You can also specify the limit in fraction of the total amount of memory
   # on the GPU. If you have a GPU with 2 GiB memory, the following is
   # equivalent to the above configuration.
   #   $ export CUPY_GPU_MEMORY_LIMIT="50%"

   import cupy
   print(cupy.get_default_memory_pool().get_limit())  # 1073741824

You can also set the limit (or override the value specified via the environment variable) using :meth:`cupy.cuda.MemoryPool.set_limit`.
In this way, you can use a different limit for each GPU device.

.. code-block:: py

   import cupy

   mempool = cupy.get_default_memory_pool()

   with cupy.cuda.Device(0):
       mempool.set_limit(size=1024**3)  # 1 GiB

   with cupy.cuda.Device(1):
       mempool.set_limit(size=2*1024**3)  # 2 GiB

.. note::

   CUDA allocates some GPU memory outside of the memory pool (such as CUDA context, library handles, etc.).
   Depending on the usage, such memory may take one to few hundred MiB.
   That will not be counted in the limit.

Changing Memory Pool
--------------------

You can use your own memory allocator instead of the default memory pool by passing the memory allocation function to :func:`cupy.cuda.set_allocator` / :func:`cupy.cuda.set_pinned_memory_allocator`.
The memory allocator function should take 1 argument (the requested size in bytes) and return :class:`cupy.cuda.MemoryPointer` / :class:`cupy.cuda.PinnedMemoryPointer`.

CuPy provides two such allocators for using managed memory and stream ordered memory on GPU,
see :func:`cupy.cuda.malloc_managed` and :func:`cupy.cuda.malloc_async`, respectively, for details.
To enable a memory pool backed by managed memory, you can construct a new :class:`~cupy.cuda.MemoryPool` instance with its allocator
set to :func:`~cupy.cuda.malloc_managed` as follows

.. code-block:: py

    import cupy

    # Use managed memory
    cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

Note that if you pass :func:`~cupy.cuda.malloc_managed` directly to :func:`~cupy.cuda.set_allocator` without constructing
a :class:`~cupy.cuda.MemoryPool` instance, when the memory is freed it will be released back to the system immediately,
which may or may not be desired.

Stream Ordered Memory Allocator is a new feature added since CUDA 11.2. CuPy provides an *experimental* interface to it.
Similar to CuPy's memory pool, Stream Ordered Memory Allocator also allocates/deallocates memory *asynchronously* from/to
a memory pool in a stream-ordered fashion. The key difference is that it is a built-in feature implemented in the CUDA
driver by NVIDIA, so other CUDA applications in the same process can easily allocate memory from the same pool.

To enable a memory pool that manages stream ordered memory, you can construct a new :class:`~cupy.cuda.MemoryAsyncPool`
instance:

.. code-block:: py

    import cupy

    # Use asynchronous stream ordered memory
    cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)

    # Create a custom stream
    s = cupy.cuda.Stream()

    # This would allocate memory asynchronously on stream s
    with s:
        a = cupy.empty((100,), dtype=cupy.float64)

Note that in this case we do not use the :class:`~cupy.cuda.MemoryPool` class. The :class:`~cupy.cuda.MemoryAsyncPool` takes
a different input argument from that of :class:`~cupy.cuda.MemoryPool` to indicate which pool to use.
Please refer to :class:`~cupy.cuda.MemoryAsyncPool`'s documentation for further detail.

Note that if you pass :func:`~cupy.cuda.malloc_async` directly to :func:`~cupy.cuda.set_allocator` without constructing
a :class:`~cupy.cuda.MemoryAsyncPool` instance, the device's *current* memory pool will be used.

When using stream ordered memory, it is important that you maintain a correct stream semantics yourselves using, for example,
the :class:`~cupy.cuda.Stream` and :class:`~cupy.cuda.Event` APIs (see :ref:`cuda_stream_event` for details); CuPy does not
attempt to act smartly for you. Upon deallocation, the memory is freed asynchronously either on the stream it was
allocated (first attempt), or on any current CuPy stream (second attempt). It is permitted that the stream on which the
memory was allocated gets destroyed before all memory allocated on it is freed.

In addition, applications/libraries internally use ``cudaMalloc`` (CUDA's default, synchronous allocator) could have unexpected
interplay with Stream Ordered Memory Allocator. Specifically, memory freed to the memory pool might not be immediately visible
to ``cudaMalloc``, leading to potential out-of-memory errors. In this case, you can either call :meth:`~cupy.cuda.MemoryAsyncPool.free_all_blocks()`
or just manually perform a (event/stream/device) synchronization, and retry.

Currently the :class:`~cupy.cuda.MemoryAsyncPool` interface is *experimental*. In particular, while its API is largely identical
to that of :class:`~cupy.cuda.MemoryPool`, several of the pool's methods require a sufficiently new driver (and of course, a
supported hardware, CUDA version, and platform) due to CUDA's limitation.

You can even disable the default memory pool by the code below.
Be sure to do this before any other CuPy operations.

.. code-block:: py

   import cupy

   # Disable memory pool for device memory (GPU)
   cupy.cuda.set_allocator(None)

   # Disable memory pool for pinned memory (CPU).
   cupy.cuda.set_pinned_memory_allocator(None)
