.. _overview:

Overview
========

`CuPy <https://github.com/cupy/cupy>`__ is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.
CuPy acts as a drop-in replacement to run existing NumPy/SciPy code on `NVIDIA CUDA <https://developer.nvidia.com/cuda-toolkit>`__ or `AMD ROCm <https://www.amd.com/en/graphics/servers-solutions-rocm>`__ platforms.

CuPy provides a ``ndarray``, sparse matrices, and the associated routines for GPU devices, all having the same API as NumPy and SciPy:

* **N-dimensional array** (``ndarray``): :doc:`cupy.ndarray <reference/ndarray>`

  * Data types (dtypes): boolean (``bool_``), integer (``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``, ``uint64``), float (``float16``, ``float32``, ``float64``), and complex (``complex64``, ``complex128``)
  * Supports the semantics identical to :class:`numpy.ndarray`, including basic / advanced indexing and broadcasting

* **Sparse matrices**: :doc:`cupyx.scipy.sparse <reference/scipy_sparse>`

  * 2-D sparse matrix: ``csr_matrix``, ``coo_matrix``, ``csc_matrix``, and ``dia_matrix``

* **NumPy Routines**

  * :doc:`Module-level Functions <reference/routines>` (``cupy.*``)
  * :doc:`Linear Algebra Functions <reference/linalg>` (``cupy.linalg.*``)
  * :doc:`Fast Fourier Transform <reference/fft>` (``cupy.fft.*``)
  * :doc:`Random Number Generator <reference/random>` (``cupy.random.*``)

* **SciPy Routines**

  * :doc:`Discrete Fourier Transforms <reference/scipy_fft>` (``cupyx.scipy.fft.*`` and ``cupyx.scipy.fftpack.*``)
  * :doc:`Advanced Linear Algebra <reference/scipy_linalg>` (``cupyx.scipy.linalg.*``)
  * :doc:`Multidimensional Image Processing <reference/scipy_ndimage>` (``cupyx.scipy.ndimage.*``)
  * :doc:`Sparse Matrices <reference/scipy_sparse>` (``cupyx.scipy.sparse.*``)
  * :doc:`Sparse Linear Algebra <reference/scipy_sparse_linalg>` (``cupyx.scipy.sparse.linalg.*``)
  * :doc:`Special Functions <reference/scipy_special>` (``cupyx.scipy.special.*``)
  * :doc:`Signal Processing <reference/scipy_signal>` (``cupyx.scipy.signal.*``)
  * :doc:`Statistical Functions <reference/scipy_stats>` (``cupyx.scipy.stats.*``)

Routines are backed by CUDA libraries (cuBLAS, cuFFT, cuSPARSE, cuSOLVER, cuRAND), Thrust, CUB, and cuTENSOR to provide the best performance.

It is also possible to easily implement :doc:`custom CUDA kernels <user_guide/kernel>` that work with ``ndarray`` using:

* **Kernel Templates**: Quickly define element-wise and reduction operation as a single CUDA kernel
* **Raw Kernel**: Import existing CUDA C/C++ code
* **Just-in-time Transpiler (JIT)**: Generate CUDA kernel from Python source code
* **Kernel Fusion**: Fuse multiple CuPy operations into a single CUDA kernel

CuPy can run in multi-GPU or cluster environments. The distributed communication package (:mod:`cupyx.distributed`) provides collective and peer-to-peer primitives for ``ndarray``, backed by NCCL.

For users who need more fine-grain control for performance, accessing :doc:`low-level CUDA features <user_guide/cuda_api>` are available:

* **Stream and Event**: CUDA stream and per-thread default stream are supported by all APIs
* **Memory Pool**: Customizable memory allocator with a built-in memory pool
* **Profiler**: Supports profiling code using CUDA Profiler and NVTX
* **Host API Binding**: Directly call CUDA libraries, such as NCCL, cuDNN, cuTENSOR, and cuSPARSELt APIs from Python

CuPy implements standard APIs for data exchange and interoperability, such as `DLPack <https://github.com/dmlc/dlpack>`__, `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__, ``__array_ufunc__`` (`NEP 13 <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`__), ``__array_function__`` (`NEP 18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`__), and `Array API Standard <https://data-apis.org/array-api/latest/>`__.
Thanks to these protocols, CuPy easily :doc:`integrates <user_guide/interoperability>` with NumPy, PyTorch, TensorFlow, MPI4Py, and any other libraries supporting the standard.

Under AMD ROCm environment, CuPy automatically translates all CUDA API calls to ROCm HIP (hipBLAS, hipFFT, hipSPARSE, hipRAND, hipCUB, hipThrust, RCCL, etc.), allowing code written using CuPy to run on both NVIDIA and AMD GPU without any modification.

Project Goal
------------

The goal of the CuPy project is to provide Python users GPU acceleration capabilities, without the in-depth knowledge of underlying GPU technologies.
The CuPy team focuses on providing:

* A complete NumPy and SciPy API coverage to become a full drop-in replacement, as well as advanced CUDA features to maximize the performance.
* Mature and quality library as a fundamental package for all projects needing acceleration, from a lab environment to a large-scale cluster.
