Basics of CuPy
==============

.. currentmodule:: cupy

In this section, you will learn about the following things:

* Basics of :class:`cupy.ndarray`
* The concept of *current device*
* host-device and device-device array transfer


Basics of cupy.ndarray
----------------------

CuPy is a GPU array backend that implements a subset of NumPy interface.
In the following code, ``cp`` is an abbreviation of ``cupy``, following the standard convention of abbreviating ``numpy`` as ``np``:

.. doctest::

   >>> import numpy as np
   >>> import cupy as cp

The :class:`cupy.ndarray` class is at the core of ``CuPy`` and is a replacement class for ``NumPy``'s :class:`numpy.ndarray`.

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])

``x_gpu`` above is an instance of :class:`cupy.ndarray`.
As one can see, CuPy's syntax here is identical to that of NumPy.
The main difference between :class:`cupy.ndarray` and :class:`numpy.ndarray` is that
the CuPy arrays are allocated on the *current device*, which we will talk about later.

Most of the array manipulations are also done in the way similar to NumPy.
Take the Euclidean norm (a.k.a L2 norm), for example.
NumPy has :func:`numpy.linalg.norm` function that calculates it on CPU.

.. doctest::

   >>> x_cpu = np.array([1, 2, 3])
   >>> l2_cpu = np.linalg.norm(x_cpu)

Using CuPy, we can perform the same calculations on GPU in a similar way:

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])
   >>> l2_gpu = cp.linalg.norm(x_gpu)

CuPy implements many functions on :class:`cupy.ndarray` objects.
See the :ref:`reference <cupy_reference>` for the supported subset of NumPy API.
Knowledge of NumPy will help you utilize most of the CuPy features.
We, therefore, recommend you familiarize yourself with the `NumPy documentation <https://docs.scipy.org/doc/numpy/index.html>`_.


Current Device
--------------

CuPy has a concept of a *current device*, which is the default GPU device on which
the allocation, manipulation, calculation, etc., of arrays take place.
Suppose ID of the current device is 0.
In such a case, the following code would create an array ``x_on_gpu0`` on GPU 0.

.. doctest::

   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

The current device can be changed using :class:`cupy.cuda.Device.use()` as follows:

.. doctest::

   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
   >>> cp.cuda.Device(1).use()
   >>> x_on_gpu1 = cp.array([1, 2, 3, 4, 5])

To temporarily switch to another GPU device, use ``with`` context manager:

.. doctest::

   >>> with cp.cuda.Device(1):
   ...    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])
   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

Most CuPy operations are performed on the currently active device and
attempts to process an array stored on a different device will result in an error:

.. doctest::

   >>> with cp.cuda.Device(0):
   ...    x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
   >>> with cp.cuda.Device(1):
   ...    x_on_gpu0 * 2  # raises error
   Traceback (most recent call last):
   ...
   ValueError: Array device must be same as the current device: array device = 0 while current = 1

``cupy.ndarray.device`` attribute indicates the device on which the array is allocated.

.. doctest::

   >>> with cp.cuda.Device(1):
   ...    x = cp.array([1, 2, 3, 4, 5])
   >>> x.device
   <CUDA Device 1>

.. note::

   When only one device is available, explicit device switching is not needed.


.. _current_stream:

Current Stream
--------------

Associated with the concept of current devices are *current streams*, which help avoid explicitly passing streams
in every single operation so as to keep the APIs pythonic and user-friendly. In CuPy, any CUDA operations
such as data transfer (see the next section) and kernel launches are enqueued onto the current stream,
and the queued tasks on the same stream will be executed in serial (but *asynchronously* with respect to the host).

The default current stream in CuPy is CUDA's null stream (i.e., stream 0). It is also known as the *legacy*
default stream, which is unique per device. However, it is possible to change the current stream using the
:class:`cupy.cuda.Stream` API, please see :doc:`cuda_api` for example. The current stream in CuPy can be
retrieved using :func:`cupy.cuda.get_current_stream`.

It is worth noting that CuPy's current stream is managed on a *per thread, per device* basis, meaning that on different
Python threads or different devices the current stream (if not the null stream) can be different.


Data Transfer
-------------

Move arrays to a device
~~~~~~~~~~~~~~~~~~~~~~~

:func:`cupy.asarray` can be used to move a :class:`numpy.ndarray`, a list, or any object
that can be passed to :func:`numpy.array` to the current device:

.. doctest::

   >>> x_cpu = np.array([1, 2, 3])
   >>> x_gpu = cp.asarray(x_cpu)  # move the data to the current device.

:func:`cupy.asarray` can accept :class:`cupy.ndarray`, which means we can
transfer the array between devices with this function.

.. doctest::

   >>> with cp.cuda.Device(0):
   ...     x_gpu_0 = cp.ndarray([1, 2, 3])  # create an array in GPU 0
   >>> with cp.cuda.Device(1):
   ...     x_gpu_1 = cp.asarray(x_gpu_0)  # move the array to GPU 1

.. note::

   :func:`cupy.asarray` does not copy the input array if possible.
   So, if you put an array of the current device, it returns the input object itself.

   If we do copy the array in this situation, you can use :func:`cupy.array` with `copy=True`.
   Actually :func:`cupy.asarray` is equivalent to `cupy.array(arr, dtype, copy=False)`.

Move array from a device to the host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving a device array to the host can be done by :func:`cupy.asnumpy` as follows:

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])  # create an array in the current device
   >>> x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.

We can also use :meth:`cupy.ndarray.get()`:

.. doctest::

   >>> x_cpu = x_gpu.get()


Memory management
-----------------

Check :doc:`./memory` for a detailed description of how is memory managed in CuPy
using memory pools.


How to write CPU/GPU agnostic code
----------------------------------

The compatibility of CuPy with NumPy enables us to write CPU/GPU generic code.
It can be made easy by the :func:`cupy.get_array_module` function.
This function returns the :mod:`numpy` or :mod:`cupy` module based on arguments.
A CPU/GPU generic function is defined using it like follows:

.. doctest::

   >>> # Stable implementation of log(1 + exp(x))
   >>> def softplus(x):
   ...     xp = cp.get_array_module(x)
   ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

Sometimes, an explicit conversion to a host or device array may be required.
:func:`cupy.asarray` and :func:`cupy.asnumpy` can be used in agnostic implementations
to get host or device arrays from either CuPy or NumPy arrays.

.. doctest::

   >>> y_cpu = np.array([4, 5, 6])
   >>> x_cpu + y_cpu
   array([5, 7, 9])
   >>> x_gpu + y_cpu
   Traceback (most recent call last):
   ...
   TypeError: Unsupported type <class 'numpy.ndarray'>
   >>> cp.asnumpy(x_gpu) + y_cpu
   array([5, 7, 9])
   >>> cp.asnumpy(x_gpu) + cp.asnumpy(y_cpu)
   array([5, 7, 9])
   >>> x_gpu + cp.asarray(y_cpu)
   array([5, 7, 9])
   >>> cp.asarray(x_gpu) + cp.asarray(y_cpu)
   array([5, 7, 9])
