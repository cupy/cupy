Fast-Fourier Transform with CuPy
================================

CuPy covers the full Fast Fourier Transform (FFT) functionalities provided in NumPy (see :doc:`../reference/fft`)
and a subset in SciPy (see :doc:`../reference/scipy_fft`). In addition to those high-level APIs that can be used
as is, CuPy provides additional features to 1. access advanced routines that `cuFFT`_ offers, 2. control better
the performance and behavior of the FFT routines. Some of these features are *experimental* (subject to change or
deprecation) or may not be applicable to `hipFFT`_/`rocFFT`_ targeting AMD GPUs.

.. _cuFFT: https://docs.nvidia.com/cuda/cufft/index.html
.. _hipFFT: https://hipfft.readthedocs.io/en/latest/
.. _rocFFT: https://rocfft.readthedocs.io/en/latest/

User-managed FFT plans
----------------------

Under construction.


FFT plan cache
--------------

Under construction.


Multi-GPU FFT
-------------

CuPy currently provides two kinds of *experimental* support for multi-GPU FFT.

.. warning::

    Using multiple GPUs to perform FFT is not guaranteed to be more performant. The rule of thumb is if the transform fits in 1 GPU, you should avoid using multiple.

The first kind of support is with the high-level :func`:`~cupy.fft.fft` and :func`:`~cupy.fft.ifft` APIs, which requires the input array to reside on one of the participating GPUs. The multi-GPU calculation is done under the hood, and by the end of the calculation the result again resides on the device where it started. Currently only 1D complex-to-complex (C2C) transform is supported; complex-to-real (C2R) or real-to-complex (R2C) transforms (such as :func:`~cupy.fft.rfft` and friends) are not. The transform can be batched or not batched (batch size = 1).

.. code:: python
    import cupy as cp
    
    cp.fft.config.use_multi_gpus = True
    cp.fft.config.set_cufft_gpus([0, 1])  # use GPU 0 & 1
    
    shape = (64, 64)
    dtype = cp.complex64
    a = cp.random.random(shape).astype(dtype)  # reside on GPU 0
    
    b = cp.fft.fft(a)  # computed on GPU 0 & 1, reside on GPU 0

If you need to perform 2D/3D transforms (ex: :func:`~cupy.fft.fftn`) instead of 1D (ex: `fft`), it would likely still work, but in this particular use case it loops over the transformed axes under the hood (which is exactly what is done in NumPy too), which could be suboptimal.

The second kind of usage is to use the low-level, *internal* CuPy APIs. You need to construct a :class:`~cupy.cuda.fft.Plan1d` object and use it as if you are programming in C/C++ with `cuFFT`_. Using this approach, your array can reside on the host as a `numpy.ndarray`, so its size can be much larger than what a single GPU can accommodate, which is one of the main reasons to run multi-GPU FFT.

.. code:: python
    import numpy as np
    import cupy as cp
    
    # no need to touch cp.fft.config, as we are using low-level API
    
    shape = (64, 64)
    dtype = np.complex64
    a = np.random.random(shape).astype(dtype)  # reside on CPU
    
    if len(shape) == 1:
        batch = 1
        nx = shape[0]
    elif len(shape) == 2:
        batch = shape[0]
        nx = shape[1]
    
    # compute via cuFFT
    cufft_type = cp.cuda.cufft.CUFFT_C2C  # single-precision c2c
    plan = cp.cuda.cufft.Plan1d(nx, cufft_type, batch, devices=[0,1])
    out_cp = np.empty_like(a)  # output on CPU
    plan.fft(a, out_cp, cufft.CUFFT_FORWARD)
    
    out_np = numpy.fft.fft(a)  # use NumPy's fft
    # np.fft.fft alway returns np.complex128
    if dtype is numpy.complex64:
        out_np = out_np.astype(dtype)
    
    # check result
    assert np.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)

For this use case, please consult the `cuFFT`_ documentation on multi-GPU transform for further detail.


Half-precision FFT
------------------

Under construction.
