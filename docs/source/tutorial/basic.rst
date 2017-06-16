Basics of CuPy
--------------

.. currentmodule:: cupy

In this section, you will learn about the following things:

* Basics of :class:`cupy.ndarray`
* The concept of *current device*
* host-device and device-device array transfer

Basics of cupy.ndarray
~~~~~~~~~~~~~~~~~~~~~~

CuPy is a GPU array backend that implements a subset of NumPy interface.
In the following code, cp is an abbreviation of cupy, as np is numpy as is customarily done:

.. doctest::

   >>> import numpy as np
   >>> import cupy as cp


The :class:`cupy.ndarray` class is in its core, which is a compatible GPU alternative of :class:`numpy.ndarray`.

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])

``x_gpu`` in the above example is an instance of :class:`cupy.ndarray`.
You can see its creation of identical to ``NumPy``'s one, except that ``numpy`` is replaced with ``cupy``.
The main difference of :class:`cupy.ndarray` from :class:`numpy.ndarray` is that the content is allocated on the device memory.
Its data is allocated on the *current device*, which will be explained later.


Most of array manipulations are also do in the way similar to NumPy.
Take the Euclid norm (a.k.a L2 norm) for example.
NumPy has `numpy.lina.g.norm` to calculate it on CPU.

.. doctest::

   >>> x_cpu = np.array([1, 2, 3])
   >>> l2_cpu = np.linalg.norm(x_cpu)

We can calculate it on GPU with CuPy in a similar way:

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])
   >>> l2_gpu = cp.linalg.norm(x_gpu)


CuPy implements many functions on :class:`cupy.ndarray` objects.
See the :ref:`reference <cupy_reference>` for the supported subset of NumPy API.
Understanding NumPy might help utilizing most features of CuPy.
So, we recommend you to read the `NumPy documentation <http://docs.scipy.org/doc/numpy/index.html>`_.


Current Device
~~~~~~~~~~~~~~

CuPy has a concept of the *current device*, which is the default device on which
the allocation, manipulation, calculation etc. of arrays are taken place.
Suppose the ID of current device is 0.
The following code allocates array contents on GPU 0.

.. doctest::

   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])


The current device can be changed by :class:`cupy.cuda.Device.use()` as follows:

.. doctest::

   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
   >>> cp.cuda.Device(1).use()
   >>> x_on_gpu1 = cp.array([1, 2, 3, 4, 5])

If you switch the current GPU temporarily, *with* statement comes in handy.

.. doctest::

   >>> with cp.cuda.Device(1):
   ...    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])
   >>> x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

Most operations of CuPy is done on the current device.
Be careful that if processing of an array on a non-current device will cause an error:

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

   If the environment has only one device, such explicit device switching is not needed.


Data Transfer
~~~~~~~~~~~~~

Move arrays to a device
-----------------------

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
------------------------------------

Moving a device array to the host can be done by :func:`cupy.asnumpy` as follows:

.. doctest::

   >>> x_gpu = cp.array([1, 2, 3])  # create an array in the current device
   >>> x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.

We can also use :meth:`cupy.ndarray.get()`:

.. doctest::

   >>> x_cpu = x_gpu.get()

.. note::

   If you work with Chainer, you can also use :func:`~chainer.cuda.to_cpu` and
   :func:`~chainer.cuda.to_gpu` to move arrays back and forth between
   a device and a host, or between different devices.
   Note that :func:`~chainer.cuda.to_gpu` has ``device`` option to specify
   the device which arrays are transferred.

How to write CPU/GPU agnostic code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compatibility of CuPy with NumPy enables us to write CPU/GPU generic code.
It can be made easy by the :func:`cupy.get_array_module` function.
This function returns the :mod:`numpy` or :mod:`cupy` module based on arguments.
A CPU/GPU generic function is defined using it like follows:

.. doctest::

   >>> # Stable implementation of log(1 + exp(x))
   >>> def softplus(x):
   ...     xp = cp.get_array_module(x)
   ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))
