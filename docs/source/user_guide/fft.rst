Fast Fourier Transform with CuPy
================================

CuPy covers the full Fast Fourier Transform (FFT) functionalities provided in NumPy (:mod:`cupy.fft`) and a
subset in SciPy (:mod:`cupyx.scipy.fft`). In addition to those high-level APIs that can be used
as is, CuPy provides additional features to

1. access advanced routines that `cuFFT`_ offers for NVIDIA GPUs,
2. control better the performance and behavior of the FFT routines.

Some of these features are *experimental* (subject to change, deprecation, or removal, see :doc:`./compatibility`)
or may be absent in `hipFFT`_/`rocFFT`_ targeting AMD GPUs.

.. _cuFFT: https://docs.nvidia.com/cuda/cufft/index.html
.. _hipFFT: https://hipfft.readthedocs.io/en/latest/
.. _rocFFT: https://rocfft.readthedocs.io/en/latest/


.. _scipy_fft_backend:

SciPy FFT backend
-----------------

Since SciPy v1.4, a backend mechanism is provided so that users can register different FFT backends and use SciPy's API to perform the actual transform
with the target backend, such as CuPy's :mod:`cupyx.scipy.fft` module. For a one-time only usage, a context manager :func:`scipy.fft.set_backend` can be used:

.. code-block:: python

    import cupy as cp
    import cupyx.scipy.fft as cufft
    import scipy.fft

    a = cp.random.random(100).astype(cp.complex64)
    with scipy.fft.set_backend(cufft):
        b = scipy.fft.fft(a)  # equivalent to cufft.fft(a)

However, such usage can be tedious. Alternatively, users can register a backend through :func:`scipy.fft.register_backend` or :func:`scipy.fft.set_global_backend`
to avoid using context managers:

.. code-block:: python

    import cupy as cp
    import cupyx.scipy.fft as cufft
    import scipy.fft
    scipy.fft.set_global_backend(cufft)

    a = cp.random.random(100).astype(cp.complex64)
    b = scipy.fft.fft(a)  # equivalent to cufft.fft(a)

.. note::

    Please refer to `SciPy FFT documentation`_ for further information.

.. note::
    To use the backend together with an explicit ``plan`` argument requires SciPy version 1.5.0 or higher.
    See below for how to create FFT plans.

.. _SciPy FFT documentation: https://docs.scipy.org/doc/scipy/reference/fft.html#backend-control


User-managed FFT plans
----------------------

For performance reasons, users may wish to create, reuse, and manage the FFT plans themselves. CuPy provides a high-level *experimental* API :func:`~cupyx.scipy.fftpack.get_fft_plan` for this need. Users specify the transform to be performed as they would with most of the high-level FFT APIs, and a plan will be generated based on the input.

.. code-block:: python

    import cupy as cp
    from cupyx.scipy.fft import get_fft_plan

    a = cp.random.random((4, 64, 64)).astype(cp.complex64)
    plan = get_fft_plan(a, axes=(1, 2), value_type='C2C')  # for batched, C2C, 2D transform

The returned plan can be used either explicitly as an argument with the :mod:`cupyx.scipy.fft` APIs:

.. code-block:: python

    import cupyx.scipy.fft

    # the rest of the arguments must match those used when generating the plan
    out = cupyx.scipy.fft.fft2(a, axes=(1, 2), plan=plan)

or as a context manager for the :mod:`cupy.fft` APIs:

.. code-block:: python

    with plan:
        # the arguments must match those used when generating the plan
        out = cp.fft.fft2(a, axes=(1, 2))


.. _fft_plan_cache:

FFT plan cache
--------------

However, there are occasions when users may *not* want to manage the FFT plans by themselves. Moreover, plans could also be reused internally in CuPy's routines, to which user-managed plans would not be applicable. Therefore, starting CuPy v8 we provide a built-in plan cache, enabled by default. The plan cache is done on a *per device, per thread* basis, and can be retrieved by the :func:`~cupy.fft.config.get_plan_cache` API.

.. code-block:: python

    >>> import cupy as cp
    >>>
    >>> cache = cp.fft.config.get_plan_cache()
    >>> cache.show_info()
    ------------------- cuFFT plan cache (device 0) -------------------
    cache enabled? True
    current / max size   : 0 / 16 (counts)
    current / max memsize: 0 / (unlimited) (bytes)
    hits / misses: 0 / 0 (counts)
    
    cached plans (most recently used first):
    
    >>> # perform a transform, which would generate a plan and cache it
    >>> a = cp.random.random((4, 64, 64))
    >>> out = cp.fft.fftn(a, axes=(1, 2))
    >>> cache.show_info()  # hit = 0
    ------------------- cuFFT plan cache (device 0) -------------------
    cache enabled? True
    current / max size   : 1 / 16 (counts)
    current / max memsize: 262144 / (unlimited) (bytes)
    hits / misses: 0 / 1 (counts)
    
    cached plans (most recently used first):
    key: ((64, 64), (64, 64), 1, 4096, (64, 64), 1, 4096, 105, 4, 'C', 2, None), plan type: PlanNd, memory usage: 262144
    
    >>> # perform the same transform again, the plan is looked up from cache and reused
    >>> out = cp.fft.fftn(a, axes=(1, 2))
    >>> cache.show_info()  # hit = 1
    ------------------- cuFFT plan cache (device 0) -------------------
    cache enabled? True
    current / max size   : 1 / 16 (counts)
    current / max memsize: 262144 / (unlimited) (bytes)
    hits / misses: 1 / 1 (counts)
    
    cached plans (most recently used first):
    key: ((64, 64), (64, 64), 1, 4096, (64, 64), 1, 4096, 105, 4, 'C', 2, None), plan type: PlanNd, memory usage: 262144
    
    >>> # clear the cache
    >>> cache.clear()
    >>> cp.fft.config.show_plan_cache_info()  # = cache.show_info(), for all devices
    =============== cuFFT plan cache info (all devices) ===============
    ------------------- cuFFT plan cache (device 0) -------------------
    cache enabled? True
    current / max size   : 0 / 16 (counts)
    current / max memsize: 0 / (unlimited) (bytes)
    hits / misses: 0 / 0 (counts)
    
    cached plans (most recently used first):
    

The returned :class:`~cupy.fft._cache.PlanCache` object has other methods for finer control, such as setting the cache size (either by counts or by memory usage). If the size is set to 0, the cache is disabled. Please refer to its documentation for more detail.

.. note::

    As shown above each FFT plan has an associated working area allocated. If an out-of-memory error happens, one may want to inspect, clear, or limit the plan cache.

.. note::

    The plans returned by :func:`~cupyx.scipy.fftpack.get_fft_plan` are not cached.


FFT callbacks
-------------

`cuFFT`_ provides FFT callbacks for merging pre- and/or post- processing kernels with the FFT routines so as to reduce the access to global memory.
This capability is supported *experimentally* by CuPy. Users need to supply custom load and/or store kernels as strings, and set up a context manager
via :func:`~cupy.fft.config.set_cufft_callbacks`. Note that the load (store) kernel pointer has to be named as ``d_loadCallbackPtr`` (``d_storeCallbackPtr``).

.. code-block:: python

    import cupy as cp

    # a load callback that overwrites the input array to 1
    code = r'''
    __device__ cufftComplex CB_ConvertInputC(
        void *dataIn,
        size_t offset,
        void *callerInfo,
        void *sharedPtr)
    {
        cufftComplex x;
        x.x = 1.;
        x.y = 0.;
        return x;
    }
    __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
    '''

    a = cp.random.random((64, 128, 128)).astype(cp.complex64)

    # this fftn call uses callback
    with cp.fft.config.set_cufft_callbacks(cb_load=code):
        b = cp.fft.fftn(a, axes=(1,2))

    # this does not use
    c = cp.fft.fftn(cp.ones(shape=a.shape, dtype=cp.complex64), axes=(1,2))

    # result agrees
    assert cp.allclose(b, c)

    # "static" plans are also cached, but are distinct from their no-callback counterparts
    cp.fft.config.get_plan_cache().show_info()


.. note::

    Internally, this feature requires recompiling a Python module *for each distinct pair* of load and store kernels. Therefore, the first invocation will be very slow, and this cost is amortized if the callbacks can be reused in the subsequent calculations. The compiled modules are cached on disk, with a default position ``$HOME/.cupy/callback_cache`` that can be changed by the environment variable ``CUPY_CACHE_DIR``.


Multi-GPU FFT
-------------

CuPy currently provides two kinds of *experimental* support for multi-GPU FFT.

.. warning::

    Using multiple GPUs to perform FFT is not guaranteed to be more performant. The rule of thumb is if the transform fits in 1 GPU, you should avoid using multiple.

The first kind of support is with the high-level :func:`~cupy.fft.fft` and :func:`~cupy.fft.ifft` APIs, which requires the input array to reside on one of the participating GPUs. The multi-GPU calculation is done under the hood, and by the end of the calculation the result again resides on the device where it started. Currently only 1D complex-to-complex (C2C) transform is supported; complex-to-real (C2R) or real-to-complex (R2C) transforms (such as :func:`~cupy.fft.rfft` and friends) are not. The transform can be either batched (batch size > 1) or not (batch size = 1).

.. code-block:: python

    import cupy as cp

    cp.fft.config.use_multi_gpus = True
    cp.fft.config.set_cufft_gpus([0, 1])  # use GPU 0 & 1

    shape = (64, 64)  # batch size = 64
    dtype = cp.complex64
    a = cp.random.random(shape).astype(dtype)  # reside on GPU 0

    b = cp.fft.fft(a)  # computed on GPU 0 & 1, reside on GPU 0

If you need to perform 2D/3D transforms (ex: :func:`~cupy.fft.fftn`) instead of 1D (ex: :func:`~cupy.fft.fft`), it would likely still work, but in this particular use case it loops over the transformed axes under the hood (which is exactly what is done in NumPy too), which could lead to suboptimal performance.

The second kind of usage is to use the low-level, *private* CuPy APIs. You need to construct a :class:`~cupy.cuda.cufft.Plan1d` object and use it as if you are programming in C/C++ with `cuFFT`_. Using this approach, your input array can reside on the host as a :class:`numpy.ndarray` so that its size can be much larger than what a single GPU can accommodate, which is one of the main reasons to run multi-GPU FFT.

.. code-block:: python

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
    # np.fft.fft always returns np.complex128
    if dtype is numpy.complex64:
        out_np = out_np.astype(dtype)

    # check result
    assert np.allclose(out_cp, out_np, rtol=1e-4, atol=1e-7)

For this use case, please consult the `cuFFT`_ documentation on multi-GPU transform for further detail.

.. note::

    The multi-GPU plans are cached if auto-generated via the high-level APIs, but not if manually generated via the low-level APIs.


Half-precision FFT
------------------

`cuFFT`_ provides ``cufftXtMakePlanMany`` and ``cufftXtExec`` routines to support a wide range of FFT needs, including 64-bit indexing and half-precision FFT. CuPy provides an *experimental* support for this capability via the new (though *private*) :class:`~cupy.cuda.cufft.XtPlanNd` API. For half-precision FFT, on supported hardware it can be twice as fast than its single-precision counterpart. NumPy does not yet provide the necessary infrastructure for half-precision complex numbers (i.e., ``numpy.complex32``), though, so the steps for this feature is currently a bit more involved than common cases.

.. code-block:: python

    import cupy as cp
    import numpy as np


    shape = (1024, 256, 256)  # input array shape
    idtype = odtype = edtype = 'E'  # = numpy.complex32 in the future

    # store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
    a = cp.random.random((shape[0], shape[1], 2*shape[2])).astype(cp.float16)
    out = cp.empty_like(a)

    # FFT with cuFFT
    plan = cp.cuda.cufft.XtPlanNd(shape[1:],
                                  shape[1:], 1, shape[1]*shape[2], idtype,
                                  shape[1:], 1, shape[1]*shape[2], odtype,
                                  shape[0], edtype,
                                  order='C', last_axis=-1, last_size=None)

    plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)

    # FFT with NumPy
    a_np = cp.asnumpy(a).astype(np.float32)  # upcast
    a_np = a_np.view(np.complex64)
    out_np = np.fft.fftn(a_np, axes=(-2,-1))
    out_np = np.ascontiguousarray(out_np).astype(np.complex64)  # downcast
    out_np = out_np.view(np.float32)
    out_np = out_np.astype(np.float16)

    # don't worry about accruacy for now, as we probably lost a lot during casting
    print('ok' if cp.mean(cp.abs(out - cp.asarray(out_np))) < 0.1 else 'not ok')

The 64-bit indexing support for all high-level FFT APIs is planned for a future CuPy release.
