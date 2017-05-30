CuPy Overview
=============

.. module:: cupy

`CuPy <https://github.com/cupy/cupy>`_ is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of :class:`cupy.ndarray`, the core multi-dimensional array class,
and many functions on it. It supports a subset of :class:`numpy.ndarray`
interface.

The following is a brief overview of supported subset of NumPy interface:

- `Basic and advanced indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
- Data types (dtypes): ``bool_``, ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``float16``, ``float32``, ``float64``
- Most of the `array creation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html>`_ (\ ``empty``, ``ones_like``, ``diag``, etc.)
- Most of the `array manipulation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html>`_ (\ ``reshape``, ``rollaxis``, ``concatenate``, etc.)
- All operators with `broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
- All `universal functions <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
  for elementwise operations (except those for complex numbers).
- `Linear algebra functions <https://docs.scipy.org/doc/numpy/reference/routines.linalg.html>`_, including product (\ ``dot``, ``matmul``, etc.) and decomposition (\ ``cholesky``, ``svd``, etc.), accelerated by `cuBLAS <https://developer.nvidia.com/cublas>`_.
- Reduction along axes (``sum``, ``max``, ``argmax``, etc.)

CuPy also includes the following features for performance:

- User-defined elementwise CUDA kernels
- User-defined reduction CUDA kernels
- Fusing CUDA kernels to optimize user-defined calculation
- Customizable memory allocator and memory pool
- `cuDNN <https://developer.nvidia.com/cudnn>`_ utilities

CuPy uses on-the-fly kernel synthesis: when a kernel call is required, it
compiles a kernel code optimized for the shapes and dtypes of given arguments,
sends it to the GPU device, and executes the kernel. The compiled code is
cached to ``$(HOME)/.cupy/kernel_cache`` directory (this cache path can be
overwritten by setting the ``CUPY_CACHE_DIR`` environment variable). It may
make things slower at the first kernel call, though this slow down will be
resolved at the second execution. CuPy also caches the kernel code sent to GPU
device within the process, which reduces the kernel transfer time on further
calls.
