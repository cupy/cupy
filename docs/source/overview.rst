.. _cupy-overview

CuPy Overview
=============

.. module:: cupy

CuPy is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, :class:`cupy.ndarray`,
and many functions on it. It supports a subset of :class:`numpy.ndarray`
interface that is enough for `Chainer <http://chainer.org/>`_.

The following is a brief overview of supported subset of NumPy interface:

- `Basic indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis)
- Element types (dtypes): bool\_, (u)int{8, 16, 32, 64}, float{16, 32, 64}
- Most of the array creation routines
- Reshaping and transposition
- All operators with broadcasting
- All `Universal functions <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ (a.k.a. ufuncs)
  for elementwise operations except those for complex numbers
- Dot product functions (except einsum) using cuBLAS
- Reduction along axes (sum, max, argmax, etc.)

CuPy also includes following features for performance:

- Customizable memory allocator, and a simple memory pool as an example
- User-defined elementwise kernels
- User-defined reduction kernels
- cuDNN utilities

CuPy uses on-the-fly kernel synthesis: when a kernel call is required, it
compiles a kernel code optimized for the shapes and dtypes of given arguments,
sends it to the GPU device, and executes the kernel. The compiled code is
cached to ``$(HOME)/.cupy/kernel_cache`` directory (this cache path can be
overwritten by setting the ``CUPY_CACHE_DIR`` environment variable). It may
make things slower at the first kernel call, though this slow down will be
resolved at the second execution. CuPy also caches the kernel code sent to GPU
device within the process, which reduces the kernel transfer time on further
calls.