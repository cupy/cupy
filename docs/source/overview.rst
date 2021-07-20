.. _overview:

Overview
========

.. currentmodule:: cupy

`CuPy <https://github.com/cupy/cupy>`_ is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of :class:`cupy.ndarray`, the core multi-dimensional array class,
and many functions on it. It supports a subset of :class:`numpy.ndarray`
interface.

The following is a brief overview of supported subset of NumPy interface:

- `Basic indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis)
- Most of `Advanced indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing>`_
  (except for some indexing patterns with boolean masks)
- Data types (dtypes): ``bool_``, ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``float16``, ``float32``, ``float64``, ``complex64``, ``complex128``
- Most of the `array creation routines <https://numpy.org/doc/stable/reference/routines.array-creation.html>`_ (\ ``empty``, ``ones_like``, ``diag``, etc.)
- Most of the `array manipulation routines <https://numpy.org/doc/stable/reference/routines.array-manipulation.html>`_ (\ ``reshape``, ``rollaxis``, ``concatenate``, etc.)
- All operators with `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
- All `universal functions <https://numpy.org/doc/stable/reference/ufuncs.html>`_
  for elementwise operations (except those for complex numbers)
- `Linear algebra functions <https://numpy.org/doc/stable/reference/routines.linalg.html>`_, including product (\ ``dot``, ``matmul``, etc.) and decomposition (\ ``cholesky``, ``svd``, etc.), accelerated by `cuBLAS <https://developer.nvidia.com/cublas>`_ and `cuSOLVER <https://developer.nvidia.com/cusolver>`_
- Multi-dimensional `fast Fourier transform <https://numpy.org/doc/stable/reference/routines.fft.html>`_ (FFT), accelerated by `cuFFT <https://developer.nvidia.com/cufft>`_
- Reduction along axes (``sum``, ``max``, ``argmax``, etc.)

CuPy additionally supports a subset of SciPy features:

- `Sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ and `sparse linear algebra <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_, powered by `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
- `Multi-dimensional image processing <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_
- `Signal processing <https://docs.scipy.org/doc/scipy/reference/signal.html>`_
- `Fast Fourier transform <https://docs.scipy.org/doc/scipy/reference/fft.html>`__ (FFT)
- `Linear algebra functions <https://docs.scipy.org/doc/scipy/reference/linalg.html>`__
- `Special functions <https://docs.scipy.org/doc/scipy/reference/special.html>`_
- `Statistical functions <https://docs.scipy.org/doc/scipy/reference/stats.html>`_

CuPy also includes the following features for performance:

- User-defined elementwise CUDA kernels
- User-defined reduction CUDA kernels
- Just-in-time compiler converting Python functions to CUDA kernels
- Fusing CUDA kernels to optimize user-defined calculation
- `CUB <https://github.com/NVIDIA/cub>`_/`cuTENSOR <https://developer.nvidia.com/cutensor>`_ backends for reduction and other routines
- Customizable memory allocator and memory pool
- `cuDNN <https://developer.nvidia.com/cudnn>`_ utilities
- Full coverage of `NCCL <https://developer.nvidia.com/nccl>`_ APIs

CuPy uses on-the-fly kernel synthesis: when a kernel call is required, it
compiles a kernel code optimized for the shapes and dtypes of given arguments,
sends it to the GPU device, and executes the kernel. The compiled code is
cached to ``$(HOME)/.cupy/kernel_cache`` directory (this cache path can be
overwritten by setting the ``CUPY_CACHE_DIR`` environment variable). It may
make things slower at the first kernel call, though this slow down will be
resolved at the second execution. CuPy also caches the kernel code sent to GPU
device within the process, which reduces the kernel transfer time on further
calls.
