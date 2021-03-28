.. _environment:

Environment variables
=====================

Here are the environment variables CuPy uses.

``CUDA_PATH``
  Path to the directory containing CUDA.
  The parent of the directory containing ``nvcc`` is used as default.
  When ``nvcc`` is not found, ``/usr/local/cuda`` is used.
  See :ref:`install_cuda` for details.

``CUPY_CACHE_DIR``
  Default: ``${HOME}/.cupy/kernel_cache``

  Path to the directory to store kernel cache.
  See :ref:`overview` for details.

``CUPY_CACHE_SAVE_CUDA_SOURCE``
  Default: ``0``

  If set to 1, CUDA source file will be saved along with compiled binary in the cache directory for debug purpose.
  Note: the source file will not be saved if the compiled binary is already stored in the cache.

``CUPY_CACHE_IN_MEMORY``
  Default: ``0``

  If set to 1, ``CUPY_CACHE_DIR`` and ``CUPY_CACHE_SAVE_CUDA_SOURCE`` will be ignored, and the cache is in memory.
  This environment variable allows reducing disk I/O, but is ignoed when ``nvcc`` is set to be the compiler backend.

``CUPY_DUMP_CUDA_SOURCE_ON_ERROR``
  Default: ``0``

  If set to 1, when CUDA kernel compilation fails,
  CuPy dumps CUDA kernel code to standard error.

``CUPY_CUDA_COMPILE_WITH_DEBUG``
  Default: ``0``

  If set to 1, CUDA kernel will be compiled with debug information (``--device-debug`` and ``--generate-line-info``).

``CUPY_GPU_MEMORY_LIMIT``
  Default: ``0`` (unlimited)

  The amount of memory that can be allocated for each device.
  The value can be specified in absolute bytes or fraction (e.g., ``"90%"``) of the total memory of each GPU.
  See :doc:`memory` for details.

``CUPY_SEED``
  Set the seed for random number generators.

``CUPY_EXPERIMENTAL_SLICE_COPY``
  Default: ``0``
  
  If set to 1, the following syntax is enabled:

    ``cupy_ndarray[:] = numpy_ndarray``

``CUPY_ACCELERATORS``
  Default: ``""`` (no accelerators)

  A comma-separated string of backend names (``cub`` or ``cutensor``) which indicates the acceleration backends used in CuPy operations and its priority.
  All accelerators are disabled by default.

``CUPY_TF32``
  Default: ``0``

  If set to 1, it allows CUDA libraries to use Tensor Cores TF32 compute for 32-bit floating point compute.

``CUPY_CUDA_ARRAY_INTERFACE_SYNC``
  Default: ``1``

  This controls CuPy's behavior as a Consumer.
  If set to 0, a stream synchronization will *not* be performed when a device array provided by an external library that implements the CUDA Array Interface is being consumed by CuPy.
  For more detail, see the `Synchronization`_ requirement in the CUDA Array Interface v3 documentation.

``CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION``
  Default: ``3``

  This controls CuPy's behavior as a Producer.
  If set to 2, the CuPy stream on which the data is being operated will not be exported and thus the Consumer (another library) will not perform any stream synchronization.
  For more detail, see the `Synchronization`_ requirement in the CUDA Array Interface v3 documentation.

``NVCC``
  Default: ``nvcc``

  Define the compiler to use when compiling CUDA source.
  Note that most CuPy kernels are built with NVRTC; this environment is only effective for RawKernels/RawModules with ``nvcc`` backend or when using ``cub`` as the accelerator.

``CUPY_CUDA_PER_THREAD_DEFAULT_STREAM``
  Default: ``0``

  If set to 1, CuPy will use the CUDA per-thread default stream, effectively causing each host thread to automatically execute in its own stream, unless the CUDA default (``null``) stream or a user-created stream is specified.
  If set to 0 (default), the CUDA default (``null``) stream is used, unless the per-thread default stream (``ptds``) or a user-created stream is specified.

CUDA Toolkit Environment Variables
  In addition to the environment variables listed above, as in any CUDA programs, all of the CUDA environment variables listed in the `CUDA Toolkit Documentation`_ will also be honored.

.. note::

  When ``CUPY_ACCELERATORS`` or ``NVCC`` environment variables are set, g++-6 or later is required as the runtime host compiler.
  Please refer to :ref:`install_cupy_from_source` for the details on how to install g++.

.. _CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

.. _Synchronization: https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization


For installation
----------------

These environment variables are used during installation (building CuPy from source).

``CUDA_PATH``
  See the description above.

``CUTENSOR_PATH``
  Path to the cuTENSOR root directory that contains ``lib`` and ``include`` directories. (experimental)

``NVCC``
  Define the compiler to use when compiling CUDA files.

``CUPY_INSTALL_USE_HIP``
  Default: ``0``

  Build CuPy for AMD ROCm Platform (experimental).
  For building the ROCm support, see :ref:`install_hip` for further detail.

``CUPY_NVCC_GENERATE_CODE``
  Build CuPy for a particular CUDA architecture.
  For example, ``CUPY_NVCC_GENERATE_CODE="arch=compute_60,code=sm_60"``.
  For specifying multiple archs, concatenate the ``arch=...`` strings with semicolons (``;``).
  If ``current`` is specified, then it will automatically detect the currently installed GPU architectures in build time.
  When this is not set, the default is to support all architectures.

``CUPY_NUM_BUILD_JOBS``
  Default: ``4``

  To enable or disable parallel build, sets the number of processes used to build the extensions in parallel.


``CUPY_NUM_NVCC_THREADS``
  Default: ``2``

  To enable or disable nvcc parallel compilation, sets the number of threads used to compile files using nvcc.
