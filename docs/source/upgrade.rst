=============
Upgrade Guide
=============

This page covers changes introduced in each major version that users should know when migrating from older releases.
Please see also the :ref:`compatibility_matrix` for supported environments of each major version.


CuPy v13
========

Modernized CCCL support and requirement
---------------------------------------

NVIDIA's CUDA C++ Core Libraries (CCCL) is the new home for the inter-dependent C++ libraries Thrust, CUB, and libcu++ that are shipped
with CUDA Toolkit 11.0+. To better serve our users with the latest CCCL features, improvements, and bug fixes, starting CuPy v13
we bundle CCCL in the source and binary (pip/conda) releases of CuPy. The same version of CCCL is used at both build-time (for building
CuPy) and run-time (for JIT-compiling kernels). This ensures uniform behavior, avoids surprises, and allows dual CUDA support as promised
by CCCL (currently CUDA 11 & 12), but this change leads to the following consequences distinct from the past releases:

* after the upgrade, the very first time of executing certain CuPy features may take longer than usual;
* the CCCL from any local CUDA installation is now ignored on purpose, either at build- or run- time;
* adventurous users who want to experiment with local CCCL changes need to update the CCCL submodule and build CuPy from source;

As a result of this movement, CuPy now follows the same compiler requirement as CCCL (and, in turn, CUDA Toolkit) and requires C++11 as
the lowest C++ standard. CCCL expects to move to C++17 in the near future.

Requirement Changes
-------------------

The following versions are no longer supported in CuPy v13.

* CUDA 11.1 or earlier
* cuDNN 8.7 or earlier
* cuTENSOR 1.x
    * Support for cuTENSOR 2.0 is added starting with CuPy v13, and support for cuTENSOR 1.x will be dropped.
      This is because there are significant API changes from cuTENSOR 1.x to 2.0, and from the maintenance perspective, it is not practical to support both cuTENSOR 1.x and 2.0 APIs simultaneously.
* Python 3.8 or earlier
* NumPy 1.21 or earlier
* Ubuntu 18.04

NumPy/SciPy Baseline API Update
-------------------------------

Baseline API has been bumped from NumPy 1.24 and SciPy 1.9 to NumPy 1.26 and SciPy 1.11.
CuPy v13 will follow the upstream products' specifications of these baseline versions.

Change in :func:`cupy.asnumpy`/:meth:`cupy.ndarray.get` Behavior
----------------------------------------------------------------

When transferring a CuPy array from GPU to CPU (as a NumPy array), previously the transfer could be nonblocking and not properly ordered when a non-default stream is in use,
leading to potential data race if the resulting array is modified on host immediately after the copy starts. In CuPy v13, the default
behavior is changed to be always blocking, with a new optional argument ``blocking`` added to allow the previous nonblocking behavior
if set to ``False``, in which case users are responsible for ensuring proper stream order.

Change in :meth:`cupy.array`/:meth:`cupy.asarray`/:meth:`cupy.asanyarray` Behavior
----------------------------------------------------------------------------------

When transferring a NumPy array from CPU to GPU, previously the transfer was always blocking even if the source array is backed by pinned memory.
In CuPy v13, the default behavior is changed to be asynchronous if the source array is allocated as pinned to improve the performance.

A new optional argument ``blocking`` has been added to allow the previous blocking behavior if set to ``True``.
You might want to set this option in case there is a possibility of overwriting the source array on CPU before the transfer completes.

Removal of ``cupy-wheel`` package
---------------------------------

The ``cupy-wheel`` package, which aimed to serve as a "meta" package that chooses and installs the right CuPy binary packages for the users' environment, has been removed in CuPy v13.
This is because the recent Pip no longer allows changing requirements dynamically.
See `#7628 <https://github.com/cupy/cupy/issues/7628>`_ for the details.

API Changes
-----------

* An *internal and undocumented* API :func:`cupy.cuda.compile_with_cache`, which was marked deprecated in CuPy v10, has been removed.
  We encourage downstream libraries and users to migrate to use public APIs, such as :class:`~cupy.RawModule` (added in CuPy v7) or :class:`~cupy.RawKernel` (added in CuPy v5).
  See :doc:`./user_guide/kernel` for their tutorials.


CUDA Runtime API is now statically linked
-----------------------------------------

CuPy is now shipped with CUDA Runtime statically linked.
Due to this, :func:`cupy.cuda.runtime.runtimeGetVersion` always returns the version of CUDA Runtime that CuPy is built with, regardless of the version of CUDA Runtime installed locally.
If you need to retrieve the version of CUDA Runtime shared library installed locally, use :func:`cupy.cuda.get_local_runtime_version` instead.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 12.2.


CuPy v12
========

Change in :class:`cupy.cuda.Device` Behavior
--------------------------------------------

The CUDA current device (set via :meth:`cupy.cuda.Device.use()` or ``cudaSetDevice()``) will be reactivated when exiting a device context manager.
This reverts the :ref:`change introduced in CuPy v10 <change in CuPy Device behavior>`, making the behavior identical to the one in CuPy v9 or earlier.

This decision was made for better interoperability with other libraries that might mutate the current CUDA device.
Suppose the following code:

.. code-block:: py

   def do_preprocess_cupy():
       with cupy.cuda.Device(2):
           # ...
           pass

   torch.cuda.set_device(1)
   do_preprocess_cupy()
   print(torch.cuda.get_device())  # -> ???

In CuPy v10 and v11, the code prints ``0``, which can be surprising for users.
In CuPy v12, the code now prints ``1``, making it easy for both users and library developers to maintain the current device where multiple devices are involved.

Deprecation of ``cupy.ndarray.scatter_{add,max,min}``
-----------------------------------------------------

These APIs have been marked as deprecated as ``cupy.{add,maximum,minimum}.at`` ufunc methods have been implemented, which behave as equivalent and NumPy-compatible.

Requirement Changes
-------------------

The following versions are no longer supported in CuPy v12.

* Python 3.7 or earlier
* NumPy 1.20 or earlier
* SciPy 1.6 or earlier

Baseline API Update
-------------------

Baseline API has been bumped from NumPy 1.23 and SciPy 1.8 to NumPy 1.24 and SciPy 1.9.
CuPy v12 will follow the upstream products' specifications of these baseline versions.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 11.8.


CuPy v11
========

Unified Binary Package for CUDA 11.2+
-------------------------------------

CuPy v11 provides a unified binary package named ``cupy-cuda11x`` that supports all CUDA 11.2+ releases.
This replaces per-CUDA version binary packages (``cupy-cuda112`` ~ ``cupy-cuda117``).

Note that CUDA 11.1 or earlier still requires per-CUDA version binary packages.
``cupy-cuda102``, ``cupy-cuda110``, and ``cupy-cuda111`` will be provided for CUDA 10.2, 11.0, and 11.1, respectively.

Requirement Changes
-------------------

The following versions are no longer supported in CuPy v11.

* ROCm 4.2 or earlier
* NumPy 1.19 or earlier
* SciPy 1.5 or earlier

CUB Enabled by Default
----------------------

CuPy v11 accelerates the computation with CUB by default.
In case needed, you can turn it off by setting :envvar:`CUPY_ACCELERATORS` environment variable to ``""``.

Baseline API Update
-------------------

Baseline API has been bumped from NumPy 1.21 and SciPy 1.7 to NumPy 1.23 and SciPy 1.8.
CuPy v11 will follow the upstream products' specifications of these baseline versions.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 11.7 and ROCm 5.0.


CuPy v10
========

Dropping CUDA 9.2 / 10.0 / 10.1 Support
---------------------------------------

CUDA 10.1 or earlier is no longer supported.
Use CUDA 10.2 or later.

Dropping NCCL v2.4 / v2.6 / v2.7 Support
----------------------------------------

NCCL v2.4, v2.6, and v2.7 are no longer supported.

Dropping Python 3.6 Support
---------------------------

Python 3.6 is no longer supported.

Dropping NumPy 1.17 Support
---------------------------

NumPy 1.17 is no longer supported.

.. _change in CuPy Device behavior:

Change in :class:`cupy.cuda.Device` Behavior
--------------------------------------------

Current device set via :meth:`~cupy.cuda.Device.use` will not be honored by the ``with Device`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   This change has been reverted in CuPy v12. See **CuPy v12** section above for details.

The current device set via :meth:`cupy.cuda.Device.use()` will not be reactivated when exiting a device context manager. An existing code mixing ``with device:`` block and ``device.use()`` may get different results between CuPy v10 and v9.

.. code-block:: py

   cupy.cuda.Device(1).use()
   with cupy.cuda.Device(0):
       pass
   cupy.cuda.Device()  # -> CuPy v10 returns device 0 instead of device 1

This decision was made to serve CuPy *users* better, but it could lead to surprises to downstream *developers* depending on CuPy,
as essentially CuPy's :class:`~cupy.cuda.Device` context manager no longer respects the CUDA ``cudaSetDevice()`` API. Mixing
device management functionalities (especially using context manager) from different libraries is highly discouraged.

For downstream libraries that still wish to respect the ``cudaGetDevice()``/``cudaSetDevice()`` APIs, you should avoid managing
current devices using the ``with Device`` context manager, and instead calling these APIs explicitly, see for example
`cupy/cupy#5963 <https://github.com/cupy/cupy/pull/5963>`_.

Changes in :class:`cupy.cuda.Stream` Behavior
---------------------------------------------

Stream is now managed per-device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previoulys, it was users' responsibility to keep the current stream consistent with the current CUDA device.
For example, the following code raises an error in CuPy v9 or earlier:

.. code-block:: py

   import cupy

   with cupy.cuda.Device(0):
       # Create a stream on device 0.
       s0 = cupy.cuda.Stream()

   with cupy.cuda.Device(1):
       with s0:
           # Try to use the stream on device 1
           cupy.arange(10)  # -> CUDA_ERROR_INVALID_HANDLE: invalid resource handle

CuPy v10 manages the current stream per-device, thus eliminating the need of switching the stream every time the active device is changed.
When using CuPy v10, the above example behaves differently because whenever a stream is created, it is automatically associated with the current device and will be ignored when switching devices. 
In early versions, trying to use `s0` in device 1 raises an error because `s0` is associated with device 0. However, in v10, this `s0` is ignored and the default stream for device 1 will be used instead.

Current stream set via ``use()`` will not be restored when exiting ``with`` block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Samely as the change of :class:`cupy.cuda.Device` above, the current stream set via :func:`cupy.cuda.Stream.use` will not be reactivated when exiting a stream context manager.
An existing code mixing ``with stream:`` block and ``stream.use()`` may get different results between CuPy v10 and v9.

.. code-block:: py

   s1 = cupy.cuda.Stream()
   s2 = cupy.cuda.Stream()
   s3 = cupy.cuda.Stream()
   with s1:
       s2.use()
       with s3:
           pass
       cupy.cuda.get_current_stream()  # -> CuPy v10 returns `s1` instead of `s2`.

Streams can now be shared between threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same :class:`cupy.cuda.Stream` instance can now safely be shared between multiple threads.

To achieve this, CuPy v10 will not destroy the stream (``cudaStreamDestroy``) if the stream is the current stream of any thread.

Big-Endian Arrays Automatically Converted to Little-Endian
----------------------------------------------------------

:func:`cupy.array`, :func:`cupy.asarray` and its variants now always transfer the data to GPU in little-endian byte order.

Previously CuPy was copying the given :class:`numpy.ndarray` to GPU as-is, regardless of the endianness.
In CuPy v10, big-endian arrays are converted to little-endian before the transfer, which is the native byte order on GPUs.
This change eliminates the need to manually change the array endianness before creating the CuPy array.

Baseline API Update
-------------------

Baseline API has been bumped from NumPy 1.20 and SciPy 1.6 to NumPy 1.21 and SciPy 1.7.
CuPy v10 will follow the upstream products' specifications of these baseline versions.

API Changes
-----------

* Device synchronize detection APIs (:func:`cupyx.allow_synchronize` and :class:`cupyx.DeviceSynchronized`), introduced as an experimental feature in CuPy v8, have been marked as deprecated because it is impossible to detect synchronizations reliably.

* An *internal* API :func:`cupy.cuda.compile_with_cache` has been marked as deprecated as there are better alternatives (see :class:`~cupy.RawModule` added since CuPy v7 and :class:`~cupy.RawKernel` since v5). While it has a longstanding history, this API has never been meant to be public. We encourage downstream libraries and users to migrate to the aforementioned public APIs. See :doc:`./user_guide/kernel` for their tutorials.

* The DLPack routine :func:`cupy.fromDlpack` is deprecated in favor of :func:`cupy.from_dlpack`, which addresses potential data race issues.

* A new module :mod:`cupyx.profiler` is added to host all profiling related APIs in CuPy. Accordingly, the following APIs are relocated to this module as follows. The old routines are deprecated.

    * :func:`cupy.prof.TimeRangeDecorator` -> :func:`cupyx.profiler.time_range`
    * :func:`cupy.prof.time_range` -> :func:`cupyx.profiler.time_range`
    * :func:`cupy.cuda.profile` -> :func:`cupyx.profiler.profile`
    * :func:`cupyx.time.repeat` -> :func:`cupyx.profiler.benchmark`

* :func:`cupy.ndarray.__pos__` now returns a copy (samely as :func:`cupy.positive`) instead of returning ``self``.

Note that deprecated APIs may be removed in the future CuPy releases.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 11.4 and ROCm 4.3.

CuPy v9
=======

Dropping Support of CUDA 9.0
----------------------------

CUDA 9.0 is no longer supported.
Use CUDA 9.2 or later.

Dropping Support of cuDNN v7.5 and NCCL v2.3
--------------------------------------------

cuDNN v7.5 (or earlier) and NCCL v2.3 (or earlier) are no longer supported.

Dropping Support of NumPy 1.16 and SciPy 1.3
--------------------------------------------

NumPy 1.16 and SciPy 1.3 are no longer supported.

Dropping Support of Python 3.5
------------------------------

Python 3.5 is no longer supported in CuPy v9.

NCCL and cuDNN No Longer Included in Wheels
-------------------------------------------

NCCL and cuDNN shared libraries are no longer included in wheels (see `#4850 <https://github.com/cupy/cupy/issues/4850>`_ for discussions). 
You can manually install them after installing wheel if you don't have a previous installation; see :doc:`install` for details.

cuTENSOR Enabled in Wheels
--------------------------

cuTENSOR can now be used when installing CuPy via wheels.

``cupy.cuda.{nccl,cudnn}`` Modules Needs Explicit Import
--------------------------------------------------------

Previously ``cupy.cuda.nccl`` and ``cupy.cuda.cudnn`` modules were automatically imported.
Since CuPy v9, these modules need to be explicitly imported (i.e., ``import cupy.cuda.nccl`` / ``import cupy.cuda.cudnn``.)

Baseline API Update
-------------------

Baseline API has been bumped from NumPy 1.19 and SciPy 1.5 to NumPy 1.20 and SciPy 1.6.
CuPy v9 will follow the upstream products' specifications of these baseline versions.

Following NumPy 1.20, aliases for the Python scalar types (``cupy.bool``, ``cupy.int``, ``cupy.float``, and ``cupy.complex``) are now deprecated.
``cupy.bool_``, ``cupy.int_``, ``cupy.float_`` and ``cupy.complex_`` should be used instead when required.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 11.2 and Python 3.8.


CuPy v8
=======

Dropping Support of CUDA 8.0 and 9.1
------------------------------------

CUDA 8.0 and 9.1 are no longer supported.
Use CUDA 9.0, 9.2, 10.0, or later.

Dropping Support of NumPy 1.15 and SciPy 1.2
--------------------------------------------

NumPy 1.15 (or earlier) and SciPy 1.2 (or earlier) are no longer supported.

Update of Docker Images
-----------------------

* CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 10.2 and Python 3.6.
* SciPy and Optuna are now pre-installed.

CUB Support and Compiler Requirement
------------------------------------

CUB module is now built by default.
You can enable the use of CUB by setting ``CUPY_ACCELERATORS="cub"`` (see :envvar:`CUPY_ACCELERATORS` for details).

Due to this change, g++-6 or later is required when building CuPy from the source.
See :doc:`install` for details.

The following environment variables are no longer effective:

* ``CUB_DISABLED``: Use :envvar:`CUPY_ACCELERATORS` as aforementioned.
* ``CUB_PATH``: No longer required as CuPy uses either the CUB source bundled with CUDA (only when using CUDA 11.0 or later) or the one in the CuPy distribution.

API Changes
-----------

* ``cupy.scatter_add``, which was deprecated in CuPy v4, has been removed. Use :func:`cupyx.scatter_add` instead.
* ``cupy.sparse`` module has been deprecated and will be removed in future releases. Use :mod:`cupyx.scipy.sparse` instead.
* ``dtype`` argument of :func:`cupy.ndarray.min` and :func:`cupy.ndarray.max` has been removed to align with the NumPy specification.
* :func:`cupy.allclose` now returns the result as 0-dim GPU array instead of Python bool to avoid device synchronization.
* :class:`cupy.RawModule` now delays the compilation to the time of the first call to align the behavior with :class:`cupy.RawKernel`.
* ``cupy.cuda.*_enabled`` flags (``nccl_enabled``, ``nvtx_enabled``, etc.) has been deprecated. Use ``cupy.cuda.*.available`` flag (``cupy.cuda.nccl.available``, ``cupy.cuda.nvtx.available``, etc.) instead.
* ``CHAINER_SEED`` environment variable is no longer effective. Use ``CUPY_SEED`` instead.


CuPy v7
=======

Dropping Support of Python 2.7 and 3.4
--------------------------------------

Starting from CuPy v7, Python 2.7 and 3.4 are no longer supported as it reaches its end-of-life (EOL) in January 2020 (2.7) and March 2019 (3.4).
Python 3.5.1 is the minimum Python version supported by CuPy v7.
Please upgrade the Python version if you are using affected versions of Python to any later versions listed under :doc:`install`.


CuPy v6
=======

Binary Packages Ignore ``LD_LIBRARY_PATH``
------------------------------------------

Prior to CuPy v6, ``LD_LIBRARY_PATH`` environment variable can be used to override cuDNN / NCCL libraries bundled in the binary distribution (also known as wheels).
In CuPy v6, ``LD_LIBRARY_PATH`` will be ignored during discovery of cuDNN / NCCL; CuPy binary distributions always use libraries that comes with the package to avoid errors caused by unexpected override.


CuPy v5
=======

``cupyx.scipy`` Namespace
-------------------------

:mod:`cupyx.scipy` namespace has been introduced to provide CUDA-enabled SciPy functions.
:mod:`cupy.sparse` module has been renamed to :mod:`cupyx.scipy.sparse`; :mod:`cupy.sparse` will be kept as an alias for backward compatibility.

Dropped Support for CUDA 7.0 / 7.5
----------------------------------

CuPy v5 no longer supports CUDA 7.0 / 7.5.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 9.2 and cuDNN 7.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.


CuPy v4
=======

.. note::

   The version number has been bumped from v2 to v4 to align with the versioning of Chainer.
   Therefore, CuPy v3 does not exist.

Default Memory Pool
-------------------

Prior to CuPy v4, memory pool was only enabled by default when CuPy is used with Chainer.
In CuPy v4, memory pool is now enabled by default, even when you use CuPy without Chainer.
The memory pool significantly improves the performance by mitigating the overhead of memory allocation and CPU/GPU synchronization.

.. attention::

   When you monitor GPU memory usage (e.g., using ``nvidia-smi``), you may notice that GPU memory not being freed even after the array instance become out of scope.
   This is expected behavior, as the default memory pool "caches" the allocated memory blocks.

To access the default memory pool instance, use :func:`get_default_memory_pool` and :func:`get_default_pinned_memory_pool`.
You can access the statistics and free all unused memory blocks "cached" in the memory pool.

.. code-block:: py

   import cupy
   a = cupy.ndarray(100, dtype=cupy.float32)
   mempool = cupy.get_default_memory_pool()

   # For performance, the size of actual allocation may become larger than the requested array size.
   print(mempool.used_bytes())   # 512
   print(mempool.total_bytes())  # 512

   # Even if the array goes out of scope, its memory block is kept in the pool.
   a = None
   print(mempool.used_bytes())   # 0
   print(mempool.total_bytes())  # 512

   # You can clear the memory block by calling `free_all_blocks`.
   mempool.free_all_blocks()
   print(mempool.used_bytes())   # 0
   print(mempool.total_bytes())  # 0

You can even disable the default memory pool by the code below.
Be sure to do this before any other CuPy operations.

.. code-block:: py

   import cupy
   cupy.cuda.set_allocator(None)
   cupy.cuda.set_pinned_memory_allocator(None)

Compute Capability
------------------

CuPy v4 now requires NVIDIA GPU with Compute Capability 3.0 or larger.
See the `List of CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_ to check if your GPU supports Compute Capability 3.0.


CUDA Stream
-----------

As CUDA Stream is fully supported in CuPy v4, ``cupy.cuda.RandomState.set_stream``, the function to change the stream used by the random number generator, has been removed.
Please use :func:`cupy.cuda.Stream.use` instead.

See the discussion in `#306 <https://github.com/cupy/cupy/pull/306>`_ for more details.

``cupyx`` Namespace
-------------------

``cupyx`` namespace has been introduced to provide features specific to CuPy (i.e., features not provided in NumPy) while avoiding collision in future.
See :doc:`reference/ext` for the list of such functions.

For this rule, :func:`cupy.scatter_add` has been moved to :func:`cupyx.scatter_add`.
:func:`cupy.scatter_add` is still available as an alias, but it is encouraged to use :func:`cupyx.scatter_add` instead.

Update of Docker Images
-----------------------

CuPy official Docker images (see :doc:`install` for details) are now updated to use CUDA 8.0 and cuDNN 6.0.
This change was introduced because CUDA 7.5 does not support NVIDIA Pascal GPUs.

To use these images, you may need to upgrade the NVIDIA driver on your host.
See `Requirements of nvidia-docker <https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements>`_ for details.

CuPy v2
=======

Changed Behavior of count_nonzero Function
------------------------------------------

For performance reasons, :func:`cupy.count_nonzero` has been changed to return zero-dimensional :class:`ndarray` instead of `int` when `axis=None`.
See the discussion in `#154 <https://github.com/cupy/cupy/pull/154>`_ for more details.


.. _compatibility_matrix:

Compatibility Matrix
====================

.. list-table::
   :header-rows: 1

   * - CuPy
     - CC [1]_
     - CUDA
     - ROCm
     - cuTENSOR
     - NCCL
     - cuDNN
     - Python
     - NumPy
     - SciPy
     - Baseline API Spec.
     - Docs
   * - v13
     - 3.5~
     - 11.2~
     - 4.3~
     - 2.0~
     - 2.16~
     - 8.8~
     - 3.9~
     - 1.22~
     - 1.7~
     - NumPy 1.26 & SciPy 1.11
     - `latest <https://docs.cupy.dev/en/latest/install.html>`__
   * - v12
     - 3.0~9.0
     - 10.2~12.x
     - 4.3 & 5.0
     - 1.4~1.7
     - 2.8~2.17
     - 7.6~8.8
     - 3.8~3.12
     - 1.21~1.26
     - 1.7~1.11
     - NumPy 1.24 & SciPy 1.9
     - `stable <https://docs.cupy.dev/en/stable/install.html>`__
   * - v11
     - 3.0~9.0
     - 10.2~12.0
     - 4.3 & 5.0
     - 1.4~1.6
     - 2.8~2.16
     - 7.6~8.7
     - 3.7~3.11
     - 1.20~1.24
     - 1.6~1.9
     - NumPy 1.23 & SciPy 1.8
     - `v11.6.0 <https://docs.cupy.dev/en/v11.6.0/install.html>`__
   * - v10
     - 3.0~8.x
     - 10.2~11.7
     - 4.0 & 4.2 & 4.3 & 5.0
     - 1.3~1.5
     - 2.8~2.11
     - 7.6~8.4
     - 3.7~3.10
     - 1.18~1.22
     - 1.4~1.8
     - NumPy 1.21 & SciPy 1.7
     - `v10.6.0 <https://docs.cupy.dev/en/v10.6.0/install.html>`__
   * - v9
     - 3.0~8.x
     - 9.2~11.5
     - 3.5~4.3
     - 1.2~1.3
     - 2.4 & 2.6~2.11
     - 7.6~8.2
     - 3.6~3.9
     - 1.17~1.21
     - 1.4~1.7
     - NumPy 1.20 & SciPy 1.6
     - `v9.6.0 <https://docs.cupy.dev/en/v9.6.0/install.html>`__
   * - v8
     - 3.0~8.x
     - 9.0 & 9.2~11.2
     - 3.x [2]_
     - 1.2
     - 2.0~2.8
     - 7.0~8.1
     - 3.5~3.9
     - 1.16~1.20
     - 1.3~1.6
     - NumPy 1.19 & SciPy 1.5
     - `v8.6.0 <https://docs.cupy.dev/en/v8.6.0/install.html>`__
   * - v7
     - 3.0~8.x
     - 8.0~11.0
     - 2.x [2]_
     - 1.0
     - 1.3~2.7
     - 5.0~8.0
     - 3.5~3.8
     - 1.9~1.19
     - (not specified)
     - (not specified)
     - `v7.8.0 <https://docs.cupy.dev/en/v7.8.0/install.html>`__
   * - v6
     - 3.0~7.x
     - 8.0~10.1
     - n/a
     - n/a
     - 1.3~2.4
     - 5.0~7.5
     - 2.7 & 3.4~3.8
     - 1.9~1.17
     - (not specified)
     - (not specified)
     - `v6.7.0 <https://docs.cupy.dev/en/v6.7.0/install.html>`__
   * - v5
     - 3.0~7.x
     - 8.0~10.1
     - n/a
     - n/a
     - 1.3~2.4
     - 5.0~7.5
     - 2.7 & 3.4~3.7
     - 1.9~1.16
     - (not specified)
     - (not specified)
     - `v5.4.0 <https://docs.cupy.dev/en/v5.4.0/install.html>`__
   * - v4
     - 3.0~7.x
     - 7.0~9.2
     - n/a
     - n/a
     - 1.3~2.2
     - 4.0~7.1
     - 2.7 & 3.4~3.6
     - 1.9~1.14
     - (not specified)
     - (not specified)
     - `v4.5.0 <https://docs.cupy.dev/en/v4.5.0/install.html>`__

.. [1] CUDA Compute Capability
.. [2] Highly experimental support with limited features.
