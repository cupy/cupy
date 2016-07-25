Environment variables
=====================

Here are the environment variables Chainer uses.


+--------------------+----------------------------------------------------+
| ``CUPY_CACHE_DIR`` | Path to the directory to store kernel cache.       |
|                    | ``$(HOME)/.cupy.kernel_cache`` is used by default. |
|                    | See :ref:`cupy-overview` for detail.               |
+--------------------+----------------------------------------------------+


For install
-----------

These environment variables are only used during installation.

+---------------+---------------------------------------------------------------------+
| ``CUDA_PATH`` | Path to the directory containing CUDA.                              |
|               | The parent of the directory containing ``nvcc`` is used as default. |
|               | When ``nvcc`` is not found, ``/usr/local/cuda`` is used.            |
|               | See :ref:`install_cuda` for details.                                |
+---------------+---------------------------------------------------------------------+
