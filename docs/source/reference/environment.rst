Environment variables
=====================

Here are the environment variables CuPy uses.


+------------------------------------+----------------------------------------------------+
| ``CUPY_CACHE_DIR``                 | Path to the directory to store kernel cache.       |
|                                    | ``$(HOME)/.cupy.kernel_cache`` is used by default. |
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


For install
-----------

These environment variables are only used during installation.

+---------------+---------------------------------------------------------------------+
| ``CUDA_PATH`` | Path to the directory containing CUDA.                              |
|               | The parent of the directory containing ``nvcc`` is used as default. |
|               | When ``nvcc`` is not found, ``/usr/local/cuda`` is used.            |
|               | See :ref:`install_cuda` for details.                                |
+---------------+---------------------------------------------------------------------+
