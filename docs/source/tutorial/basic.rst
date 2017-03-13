Basics of CuPy
--------------

.. currentmodule:: cupy

In this section, you will learn about the following things:

* Basics of `cupy.ndarray`
* The concept of *current device*
* CPU-GPU/GPU-GPU array transfer
* How to write CPU-GPU agnostic codes with CuPy


:class:`cupy.ndarray`
~~~~~~~~~~~~~~~~~~~~~

CuPy is a GPU array backend that implements a subset of NumPy interface.
The :class:`cupy.ndarray` class is in its core, which is a compatible GPU alternative of :class:`numpy.ndarray`.

.. testcode::

   x_gpu = cupy.array([1, 2, 3])

``x_gpu`` in the above example is an instane of :class:`cupy.ndarray`.
You can see its creation of identical to ``NumPy``'s one, except that ``numpy`` is replaced with ``cupy``.
The main difference of :class:`cupy.ndarray` from :class:`numpy.ndarray` is that the content is allocated on the device memory.
Its data is allocated on the *current device*, which will be explained later.


Most of array manipulations are also do in the way similar to NumPy.
Take the Euclid norm (a.k.a L2 norm) for example.
NumPy has `numpy.lina.g.norm` to calculate it on CPU.

.. testcode::

   x_cpu = numpy.array([1, 2, 3])
   l2_cpu = numpy.linalg.norm(x_cpu)

We can calculate it on GPU with CuPy in a similar way:

.. testcode::

   x_gpu = cupy.array([1, 2, 3])
   l2_gpu = cupy.linalg.norm(x_gpu)


CuPy implements many functions on :class:`cupy.ndarray` objects.
See the :ref:`reference <cupy_reference>` for the supported subset of NumPy API.
Understanding NumPy might help utilizing most features of CuPy.
So, we recommend you to read the `NumPy documentation <http://docs.scipy.org/doc/numpy/index.html>`_.


Current Device
~~~~~~~~~~~~~~

CuPy has a concept of the *current* device, which is the default device on which
the allocation, manipulation calculation and so on of arrays take place.
Suppose the ID of current device is 0.
The following code allocates array contents on GPU 0.

.. testcode::

   x_on_gpu0 = cupy.array([1, 2, 3, 4, 5])


The current device can be changed by :class:`cupy.cuda.Device.use()` as follows:

.. testcode::

   x_on_gpu0 = cupy.array([1, 2, 3, 4, 5])
   cupy.cuda.Device(1).use()
   x_on_gpu1 = cupy.array([1, 2, 3, 4, 5])

If you switch the current GPU temporarily, *with* statement comes in handy.

.. testcode::

   with cupy.cuda.Device(1):
       x_on_gpu1 = cupy.array([1, 2, 3, 4, 5])

   x_on_gpu0 = cupy.array([1, 2, 3, 4, 5])

Most operations of CuPy is done on the current device.
Be careful that if processing of an array on a non-current device will cause an error:

.. testcode::

   x_on_gpu0 = cupy.array([1, 2, 3, 4, 5])

   with cupy.cuda.Device(1):
       x_on_gpu1 * 2  # raises error


``cupy.ndarray.device`` attribute indicates the device on which the array is allocated.


.. testcode::

   with cupy.cuda.Device(1):
       x = cupy.array([1, 2, 3, 4, 5])

   x.device

.. testoutput::

   <CUDA Device 1>


.. note::

   The *with* statements in these codes are required to select the appropriate CUDA device.
   If user uses only one device, these device switching is not needed.


Data Transfer
~~~~~~~~~~~~~

:func:`cupy.cuda.to_gpu` is used to transfer data to GPU from either CPU (:class:`numpy.ndarray`)
or (possibly another) GPU (:class:`cupy.ndarray`).

.. testcode::

   x_cpu = numpy.array([1, 2, 3])
   x_gpu_0 = cupy.cuda.to_gpu(x_cpu, device=0)
   x_gpu_1 = cupy.cuda.to_gpu(x_gpu_0, device=1)

If we omit ``device`` option, the array is transferred to the current device.

:class:`cupy.ndarray` also has ``to_gpu`` method for transferring itself to another device.

.. testcode::

   with cupy.cuda.Device(1):
       x = cupy.array([1, 2, 3, 4, 5])  # x in on GPU 1

   x.to_gpu(device=0)  # x is moved to GPU 0


On the other hand, moving a device array to the host can be done by :func:`cupy.asnumpy` as follows:

.. testcode::

   x_cpu = cupy.asnumpy(x_gpu)

It is equivalent to the following code:

.. testcode::

   x_cpu = x_gpu.get()


CPU-GPU generic code
~~~~~~~~~~~~~~~~~~~~~

The APIs of CuPy is designed to align with NumPy's ones, as we saw in the previous sections.
This compatibility enables us to write *CPU/GPU generic* code, or, code that work with both on CPU and GPU.

CuPy offers an utility function :func:`cupy.get_array_module`,
that returns :mod:`numpy` or :mod:`cupy` module based on arguments.
The following is an example of CPU/GPU generic softplus function with it.

.. testcode::

   # Stable implementation of softplus(x) = log(1 + exp(x))
   def softplus(x):
       xp = cupy.get_array_module(x)
       return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

It accepts if ``x`` is either :class:`numpy.ndarray` or :class:`cupy.ndarray`
and returns the resulting array which is allocated on the same hardware as ``x``.
