Basics of CuPy
--------------

.. currentmodule:: cupy

In this section, you will learn about the following things:

* Basics of CuPy
* How to write CPU-GPU agnostic codes with CuPy


Basics of :class:`cupy.ndarray`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy is a GPU array backend that implements a subset of NumPy interface.
The :class:`cupy.ndarray` class is in its core, which is a compatible GPU alternative of :class:`numpy.ndarray`.

.. testcode::

   x_gpu = cupy.array([1, 2, 3])

`x_gpu` in the above example is an instane of :class:`cupy.ndarray`.
You can see its creation of identical to ``NumPy``'s one, except that ``numpy`` is replaced with ``cupy``.
The main difference of :class:`cupy.ndarray` from :class:`numpy.ndarray` is that the content is allocated on the device memory.
Its data is allocated on the current GPU device (the concept of *current device* will explained later).


Most of array manipulations are also do in the way similar to NumPy.
Take the Euclid norm (a.k.a L2 norm) of an array for example.
We can calculate it on CPU with NumPy as follows:

.. testcode::

   x_cpu = numpy.array([1, 2, 3])
   l2_cpu = numpy.linalg.norm(x_cpu)

We can do the same thing on GPU with CuPy in a similar way:

.. testcode::

   x_gpu = cupy.array([1, 2, 3])
   l2_gpu = cupy.linalg.norm(x_gpu)


CuPy implements many functions on :class:`cupy.ndarray` objects.
See the :ref:`reference <cupy_reference>` for the supported subset of NumPy API.
Understanding NumPy might help utilizing most features of CuPy.
See the `NumPy documentation <http://docs.scipy.org/doc/numpy/index.html>`_.


Current Device
~~~~~~~~~~~~~~

CuPy has a concept of the *current* device, which is the default device on which
the allocation, manipulation calculation and so on of arrays take place.
Suppose the ID of current device is 0.
The following code allocates array contents on GPU 0.

.. testcode::

   x_on_gpu0 = cupy.array([1, 2, 3, 4, 5])

The current device can be changed by :class:`cupy.cuda.Device` object as follows:

.. testcode::

   with cupy.cuda.Device(1):
       x_on_gpu1 = cupy.array([1, 2, 3, 4, 5])


Most operations of CuPy is done on the current device.
Be careful that it causes an error to process an array on a non-current device.

.. testcode::

   with cupy.cuda.Device(1):
       x_on_gpu1 = cupy.array([1, 2, 3, 4, 5])

   with cupy.cuda.Device(0):
       x_on_gpu1 * 2  # raises error


``cupy.ndarray.device`` attribute indicates the device on which the array is allocated.


.. testcode::

   with cupy.cuda.Device(1):
       x = cupy.array([1, 2, 3, 4, 5])

   x.device


Data Transfer
~~~~~~~~~~~~~

You can transfer array from one GPU to another GPU with `to_gpu` method.

.. testcode::

   with cupy.cuda.Device(1):
       x = cupy.array([1, 2, 3, 4, 5])  # x in on GPU 1

   x.to_gpu(device=0)  # x is moved to GPU 0


`cupy.cuda.to_gpu` is used to transfer `numpy.ndarray` to `cupy.ndarray`.

.. testcode::

   x_cpu = numpy.array([1, 2, 3])
   x_gpu = cupy.cuda.to_gpu(x_cpu, device=0)

If we omit `device` option, the array is transferred to the current device by default.

We can transfer the array from one device to another device with ``to_gpu`` method.

.. testcode::

   with cupy.cuda.Device(1):
       x = cupy.array([1, 2, 3, 4, 5])  # x in on GPU 1

   x.to_gpu(device=0)  # x is moved to GPU 0

Moving a device array to the host can be done by :func:`cupy.asnumpy` as follows:

.. testcode::

   x_cpu = cupy.asnumpy(x_gpu)

It is equivalent to the following code:

.. testcode::

   x_cpu = x_gpu.get()

.. note::

   The *with* statements in these codes are required to select the appropriate CUDA device.
   If user uses only one device, these device switching is not needed.



CPU-GPU agnostic code
~~~~~~~~~~~~~~~~~~~~~

The compatibility of CuPy with NumPy enables us to write CPU/GPU generic code.
It can be made easy by the :func:`cupy.get_array_module` function.
This function returns the :mod:`numpy` or :mod:`cupy` module based on arguments.
A CPU/GPU generic function is defined using it like follows:

.. testcode::

   # Stable implementation of log(1 + exp(x))
   def softplus(x):
       xp = cupy.get_array_module(x)
       return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))