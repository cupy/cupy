.. _environment:

Environment variables
=====================

For runtime
-----------

Here are the environment variables that CuPy uses at runtime.

.. envvar:: CUDA_PATH

  Path to the directory containing CUDA.
  The parent of the directory containing ``nvcc`` is used as default.
  When ``nvcc`` is not found, ``/usr/local/cuda`` is used.
  See :ref:`install_cuda` for details.

.. envvar:: CUPY_CACHE_DIR

  Default: ``${HOME}/.cupy/kernel_cache``

  Path to the directory to store kernel cache.
  See :doc:`../user_guide/performance` for details.

.. envvar:: CUPY_CACHE_SAVE_CUDA_SOURCE

  Default: ``0``

  If set to ``1``, CUDA source file will be saved along with compiled binary in the cache directory for debug purpose.
  Note: the source file will not be saved if the compiled binary is already stored in the cache.

.. envvar:: CUPY_CACHE_IN_MEMORY

  Default: ``0``

  If set to ``1``, :envvar:`CUPY_CACHE_DIR` and :envvar:`CUPY_CACHE_SAVE_CUDA_SOURCE` will be ignored, and the cache is in memory.
  This environment variable allows reducing disk I/O, but is ignoed when ``nvcc`` is set to be the compiler backend.

.. envvar:: CUPY_DISABLE_JITIFY_CACHE

  Default: ``0``

  If set to ``1``, headers loaded by Jitify would not be cached on disk (to :envvar:`CUPY_CACHE_DIR`). The default is to
  always cache.

.. envvar:: CUPY_DUMP_CUDA_SOURCE_ON_ERROR

  Default: ``0``

  If set to ``1``, when CUDA kernel compilation fails,
  CuPy dumps CUDA kernel code to standard error.

.. envvar:: CUPY_CUDA_COMPILE_WITH_DEBUG

  Default: ``0``

  If set to ``1``, CUDA kernel will be compiled with debug information (``--device-debug`` and ``--generate-line-info``).

.. envvar:: CUPY_GPU_MEMORY_LIMIT

  Default: ``0`` (unlimited)

  The amount of memory that can be allocated for each device.
  The value can be specified in absolute bytes or fraction (e.g., ``"90%"``) of the total memory of each GPU.
  See :doc:`../user_guide/memory` for details.

.. envvar:: CUPY_SEED

  Set the seed for random number generators.

.. envvar:: CUPY_EXPERIMENTAL_SLICE_COPY

  Default: ``0``
  
  If set to ``1``, the following syntax is enabled::

    cupy_ndarray[:] = numpy_ndarray

.. envvar:: CUPY_ACCELERATORS

  Default: ``"cub"`` (In ROCm HIP environment, the default value is ``""``. i.e., no accelerators are used.)

  A comma-separated string of backend names (``cub``, ``cutensor``, or ``cutensornet``) which indicates the acceleration backends used in CuPy operations and its priority (in descending order).
  By default, all accelerators are disabled on HIP and only CUB is enabled on CUDA.

.. envvar:: CUPY_TF32

  Default: ``0``

  If set to ``1``, it allows CUDA libraries to use Tensor Cores TF32 compute for 32-bit floating point compute.

.. envvar:: CUPY_CUDA_ARRAY_INTERFACE_SYNC

  Default: ``1``

  This controls CuPy's behavior as a Consumer.
  If set to ``0``, a stream synchronization will *not* be performed when a device array provided by an external library that implements the CUDA Array Interface is being consumed by CuPy.
  For more detail, see the `Synchronization`_ requirement in the CUDA Array Interface v3 documentation.

.. envvar:: CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION

  Default: ``3``

  This controls CuPy's behavior as a Producer.
  If set to ``2``, the CuPy stream on which the data is being operated will not be exported and thus the Consumer (another library) will not perform any stream synchronization.
  For more detail, see the `Synchronization`_ requirement in the CUDA Array Interface v3 documentation.

.. envvar:: CUPY_DLPACK_EXPORT_VERSION

  Default: ``0.6``

  This controls CuPy's DLPack support. Currently, setting a value smaller than 0.6 would disguise managed memory as normal device memory, which enables data exchanges with libraries that have not updated their DLPack support, whereas starting 0.6 CUDA managed memory can be correctly recognized as a valid device type.

.. envvar:: NVCC

  Default: ``nvcc``

  Define the compiler to use when compiling CUDA source.
  Note that most CuPy kernels are built with NVRTC; this environment variable is only effective for :class:`~cupy.RawKernel`/:class:`~cupy.RawModule` with the ``nvcc`` backend or when using ``cub`` as the accelerator.

.. envvar:: CUPY_CUDA_PER_THREAD_DEFAULT_STREAM

  Default: ``0``

  If set to ``1``, CuPy will use the CUDA per-thread default stream, effectively causing each host thread to automatically execute in its own stream, unless the CUDA default (``null``) stream or a user-created stream is specified.
  If set to ``0`` (default), the CUDA default (``null``) stream is used, unless the per-thread default stream (``ptds``) or a user-created stream is specified.

.. envvar:: CUPY_COMPILE_WITH_PTX

  Default: ``0``

  By default, CuPy directly compiles kernels into SASS (CUBIN) to support `CUDA Enhanced Compatibility <https://docs.nvidia.com/deploy/cuda-compatibility/>`_
  If set to ``1``, CuPy instead compiles kernels into PTX and lets CUDA Driver assemble SASS from PTX.
  This option is only effective for CUDA 11.1 or later; CuPy always compiles into PTX on earlier CUDA versions. Also, this option only applies when NVRTC is selected as the compilation backend. NVCC backend always compiles into SASS (CUBIN).

CUDA Toolkit Environment Variables
  In addition to the environment variables listed above, as in any CUDA programs, all of the CUDA environment variables listed in the `CUDA Toolkit Documentation`_ will also be honored.

.. note::

  When :envvar:`CUPY_ACCELERATORS` or :envvar:`NVCC` environment variables are set, g++-6 or later is required as the runtime host compiler.
  Please refer to :ref:`install_cupy_from_source` for the details on how to install g++.

.. _CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars

.. _Synchronization: https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization


For installation
----------------

These environment variables are used during installation (building CuPy from source).

.. envvar:: CUTENSOR_PATH

  Path to the cuTENSOR root directory that contains ``lib`` and ``include`` directories. (experimental)

.. envvar:: CUPY_INSTALL_USE_HIP

  Default: ``0``

  If set to ``1``, CuPy is built for AMD ROCm Platform (experimental).
  For building the ROCm support, see :ref:`install_hip` for further detail.

.. envvar:: CUPY_USE_CUDA_PYTHON

  Default: ``0``

  If set to ``1``, CuPy is built using `CUDA Python <https://github.com/NVIDIA/cuda-python>`_.

.. envvar:: CUPY_NVCC_GENERATE_CODE

  Build CuPy for a particular CUDA architecture. For example::

    CUPY_NVCC_GENERATE_CODE="arch=compute_60,code=sm_60"

  For specifying multiple archs, concatenate the ``arch=...`` strings with semicolons (``;``).
  If ``current`` is specified, then it will automatically detect the currently installed GPU architectures in build time.
  When this is not set, the default is to support all architectures.

.. envvar:: CUPY_NUM_BUILD_JOBS

  Default: ``4``

  To enable or disable parallel build, sets the number of processes used to build the extensions in parallel.


.. envvar:: CUPY_NUM_NVCC_THREADS

  Default: ``2``

  To enable or disable nvcc parallel compilation, sets the number of threads used to compile files using nvcc.

Additionally, the environment variables :envvar:`CUDA_PATH` and :envvar:`NVCC` are also respected at build time.
