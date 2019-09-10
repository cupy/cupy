.. _environment:

Environment variables
=====================

Here are the environment variables CuPy uses.


+------------------------------------+----------------------------------------------------+
| ``CUDA_PATH``                      | Path to the directory containing CUDA.             |
|                                    | The parent of the directory containing ``nvcc`` is |
|                                    | used as default.                                   |
|                                    | When ``nvcc`` is not found, ``/usr/local/cuda`` is |
|                                    | used.                                              |
|                                    | See :ref:`install_cuda` for details.               |
+------------------------------------+----------------------------------------------------+
| ``CUPY_CACHE_DIR``                 | Path to the directory to store kernel cache.       |
|                                    | ``${HOME}/.cupy/kernel_cache`` is used by default. |
|                                    | See :ref:`overview` for details.                   |
+------------------------------------+----------------------------------------------------+
| ``CUPY_CACHE_SAVE_CUDA_SOURCE``    | If set to 1, CUDA source file will be saved along  |
|                                    | with compiled binary in the cache directory for    |
|                                    | debug purpose. It is disabled by default.          |
|                                    | Note: source file will not be saved if the         |
|                                    | compiled binary is already stored in the cache.    |
+------------------------------------+----------------------------------------------------+
| ``CUPY_DUMP_CUDA_SOURCE_ON_ERROR`` | If set to 1, when CUDA kernel compilation fails,   |
|                                    | CuPy dumps CUDA kernel code to standard error.     |
|                                    | It is disabled by default.                         |
+------------------------------------+----------------------------------------------------+
| ``CUPY_CUDA_COMPILE_WITH_DEBUG``   | If set to 1, CUDA kernel will be compiled with     |
|                                    | debug information (``--device-debug`` and          |
|                                    | ``--generate-line-info``).                         |
|                                    | It is disabled by default.                         |
+------------------------------------+----------------------------------------------------+
| ``CUPY_GPU_MEMORY_LIMIT``          | The amount of memory that can be allocated for     |
|                                    | each device.                                       |
|                                    | The value can be specified in absolute bytes or    |
|                                    | fraction (e.g., ``"90%"``) of the total memory of  |
|                                    | each GPU.                                          |
|                                    | See :doc:`memory` for details.                     |
|                                    | ``0`` (unlimited) is used by default.              |
+------------------------------------+----------------------------------------------------+


For install
-----------

These environment variables are used during installation (building CuPy from source).

+-------------------+---------------------------------------------------------------------+
| ``CUDA_PATH``     | See the description above.                                          |
+-------------------+---------------------------------------------------------------------+
| ``CUTENSOR_PATH`` | Path to the cuTENSOR root directory that contains ``lib`` and       |
|                   | ``include`` directories. (experimental)                             |
+-------------------+---------------------------------------------------------------------+
| ``NVCC``          | Define the compiler to use when compiling CUDA files.               |
+-------------------+---------------------------------------------------------------------+
