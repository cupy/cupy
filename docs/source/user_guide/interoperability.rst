Interoperability
================

CuPy can also be used in conjunction with other libraries.

NumPy
-----

:class:`cupy.ndarray` implements ``__array_ufunc__`` interface (see `NEP 13 — A Mechanism for Overriding Ufuncs <http://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_ for details).
This enables NumPy ufuncs to be directly operated on CuPy arrays.
``__array_ufunc__`` feature requires NumPy 1.13 or later.

.. code:: python

    import cupy
    import numpy

    arr = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)
    result = numpy.sum(arr)
    print(type(result))  # => <class 'cupy._core.core.ndarray'>

:class:`cupy.ndarray` also implements ``__array_function__`` interface (see `NEP 18 — A dispatch mechanism for NumPy’s high level array functions <http://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ for details).
This enables code using NumPy to be directly operated on CuPy arrays.
``__array_function__`` feature requires NumPy 1.16 or later; note that this is currently defined as an experimental feature of NumPy and you need to specify the environment variable (``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1``) to enable it.


Numba
-----

`Numba <https://numba.pydata.org/>`_ is a Python JIT compiler with NumPy support.

:class:`cupy.ndarray` implements ``__cuda_array_interface__``, which is the CUDA array interchange interface compatible with Numba v0.39.0 or later (see `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ for details).
It means you can pass CuPy arrays to kernels JITed with Numba.
The following is a simple example code borrowed from `numba/numba#2860 <https://github.com/numba/numba/pull/2860>`_:

.. code:: python

	import cupy
	from numba import cuda

	@cuda.jit
	def add(x, y, out):
		start = cuda.grid(1)
		stride = cuda.gridsize(1)
		for i in range(start, x.shape[0], stride):
			out[i] = x[i] + y[i]

	a = cupy.arange(10)
	b = a * 2
	out = cupy.zeros_like(a)

	print(out)  # => [0 0 0 0 0 0 0 0 0 0]

	add[1, 32](a, b, out)

	print(out)  # => [ 0  3  6  9 12 15 18 21 24 27]

In addition, :func:`cupy.asarray` supports zero-copy conversion from Numba CUDA array to CuPy array.

.. code:: python

    import numpy
    import numba
    import cupy

    x = numpy.arange(10)  # type: numpy.ndarray
    x_numba = numba.cuda.to_device(x)  # type: numba.cuda.cudadrv.devicearray.DeviceNDArray
    x_cupy = cupy.asarray(x_numba)  # type: cupy.ndarray

.. warning::

    ``__cuda_array_interface__`` specifies that the object lifetime must be managed by the user, so it is an undefined behavior if the
    exported object is destroyed while still in use by the consumer library.

.. note::

    CuPy has a few environment variables controlling the exchange behavior, see :doc:`../reference/environment` for details.


mpi4py
------

`MPI for Python (mpi4py) <https://mpi4py.readthedocs.io/en/latest/>`_ is a Python wrapper for the Message Passing Interface (MPI) libraries.

MPI is the most widely used standard for high-performance inter-process communications. Recently several MPI vendors, including MPICH, Open MPI and MVAPICH, have extended their support beyond the MPI-3.1 standard to enable "CUDA-awareness"; that is, passing CUDA device pointers directly to MPI calls to avoid explicit data movement between the host and the device.

With the aforementioned ``__cuda_array_interface__`` standard implemented in CuPy, mpi4py now provides (experimental) support for passing CuPy arrays to MPI calls, provided that mpi4py is built against a CUDA-aware MPI implementation. The following is a simple example code borrowed from `mpi4py Tutorial <https://mpi4py.readthedocs.io/en/latest/tutorial.html>`_:

.. code:: python

    # To run this script with N MPI processes, do
    # mpiexec -n N python this_script.py

    import cupy
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # Allreduce
    sendbuf = cupy.arange(10, dtype='i')
    recvbuf = cupy.empty_like(sendbuf)
    comm.Allreduce(sendbuf, recvbuf)
    assert cupy.allclose(recvbuf, sendbuf*size)

This new feature will be officially released in mpi4py 3.1.0. To try it out, please build mpi4py from source for the time being. See the `mpi4py website <https://mpi4py.readthedocs.io/en/latest/>`_ for more information.


PyTorch
-------

`PyTorch <https://pytorch.org/>`_ is a machine learning framefork that provides high-performance, differentiable tensor operations.

PyTorch also supports ``__cuda_array_interface__``, so zero-copy data exchange between CuPy and PyTorch can be achieved at no cost.
The only caveat is PyTorch by default creates CPU tensors, which do not have the ``__cuda_array_interface__`` property defined, and
users need to ensure the tensor is already on GPU before exchanging.

.. code:: python

    >>> import cupy as cp
    >>> import torch
    >>>
    >>> # convert a torch tensor to a cupy array
    >>> a = torch.rand((4, 4), device='cuda')
    >>> b = cp.asarray(a)
    >>> b *= b
    >>> b
    array([[0.8215962 , 0.82399917, 0.65607935, 0.30354425],
           [0.422695  , 0.8367199 , 0.00208597, 0.18545236],
           [0.00226746, 0.46201342, 0.6833052 , 0.47549972],
           [0.5208748 , 0.6059282 , 0.1909013 , 0.5148635 ]], dtype=float32)
    >>> a
    tensor([[0.8216, 0.8240, 0.6561, 0.3035],
            [0.4227, 0.8367, 0.0021, 0.1855],
            [0.0023, 0.4620, 0.6833, 0.4755],
            [0.5209, 0.6059, 0.1909, 0.5149]], device='cuda:0')
    >>> # check the underlying memory pointer is the same
    >>> assert a.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]
    >>>
    >>> # convert a cupy array to a torch tensor
    >>> a = cp.arange(10)
    >>> b = torch.as_tensor(a, device='cuda')
    >>> b += 3
    >>> b
    tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12], device='cuda:0')
    >>> a
    array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
    >>> assert a.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]

PyTorch also supports zero-copy data exchange through ``DLPack`` (see :ref:`dlpack` below):

.. code:: python

	import cupy
	import torch

	from torch.utils.dlpack import to_dlpack
	from torch.utils.dlpack import from_dlpack

	# Create a PyTorch tensor.
	tx1 = torch.randn(1, 2, 3, 4).cuda()

	# Convert it into a DLPack tensor.
	dx = to_dlpack(tx1)

	# Convert it into a CuPy array.
	cx = cupy.fromDlpack(dx)

	# Convert it back to a PyTorch tensor.
	tx2 = from_dlpack(cx.toDlpack())

`pytorch-pfn-extras <https://github.com/pfnet/pytorch-pfn-extras/>`_ library provides additional integration features with PyTorch, including memory pool sharing and stream sharing:

.. code:: python

   >>> import cupy
   >>> import torch
   >>> import pytorch_pfn_extras as ppe
   >>>
   >>> # Perform CuPy memory allocation using the PyTorch memory pool.
   >>> ppe.cuda.use_torch_mempool_in_cupy()
   >>> torch.cuda.memory_allocated()
   0
   >>> arr = cupy.arange(10)
   >>> torch.cuda.memory_allocated()
   512
   >>>
   >>> # Change the default stream in PyTorch and CuPy:
   >>> stream = torch.cuda.Stream()
   >>> with ppe.cuda.stream(stream):
   ...     ...


RMM
---

`RMM (RAPIDS Memory Manager) <https://docs.rapids.ai/api/rmm/stable/index.html>`_ provides highly configurable memory allocators.

RMM provides an interface to allow CuPy to allocate memory from the RMM memory pool instead of from CuPy's own pool. It can be set up
as simple as:

.. code:: python

    import cupy
    import rmm
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

Sometimes, a more performant allocator may be desirable. RMM provides an option to switch the allocator:

.. code:: python

    import cupy
    import rmm
    rmm.reinitialize(pool_allocator=True)  # can also set init pool size etc here
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

For more information on CuPy's memory management, see :doc:`./memory`.


.. _dlpack:

DLPack
------

`DLPack <https://github.com/dmlc/dlpack>`__ is a specification of tensor structure to share tensors among frameworks.

CuPy supports importing from and exporting to DLPack data structure (:func:`cupy.fromDlpack` and :func:`cupy.ndarray.toDlpack`).

Here is a simple example:

.. code:: python

	import cupy

	# Create a CuPy array.
	cx1 = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)

	# Convert it into a DLPack tensor.
	dx = cx1.toDlpack()

	# Convert it back to a CuPy array.
	cx2 = cupy.fromDlpack(dx)

`TensorFlow <https://www.tensorflow.org>`_ also supports DLpack, so zero-copy data exchange between CuPy and TensorFlow through
DLPack is possible:

.. code:: python

    >>> import tensorflow as tf
    >>> import cupy as cp
    >>>
    >>> # convert a TF tensor to a cupy array
    >>> with tf.device('/GPU:0'):
    ...     a = tf.random.uniform((10,))
    ...
    >>> a
    <tf.Tensor: shape=(10,), dtype=float32, numpy=
    array([0.9672388 , 0.57568085, 0.53163004, 0.6536236 , 0.20479882,
           0.84908986, 0.5852566 , 0.30355775, 0.1733712 , 0.9177849 ],
          dtype=float32)>
    >>> a.device
    '/job:localhost/replica:0/task:0/device:GPU:0'
    >>> cap = tf.experimental.dlpack.to_dlpack(a)
    >>> b = cp.fromDlpack(cap)
    >>> b *= 3
    >>> b
    array([1.4949363 , 0.60699713, 1.3276931 , 1.5781245 , 1.1914308 ,
           2.3180873 , 1.9560868 , 1.3932796 , 1.9299742 , 2.5352407 ],
          dtype=float32)
    >>> a
    <tf.Tensor: shape=(10,), dtype=float32, numpy=
    array([1.4949363 , 0.60699713, 1.3276931 , 1.5781245 , 1.1914308 ,
           2.3180873 , 1.9560868 , 1.3932796 , 1.9299742 , 2.5352407 ],
          dtype=float32)>
    >>>
    >>> # convert a cupy array to a TF tensor
    >>> a = cp.arange(10)
    >>> cap = a.toDlpack()
    >>> b = tf.experimental.dlpack.from_dlpack(cap)
    >>> b.device
    '/job:localhost/replica:0/task:0/device:GPU:0'
    >>> b
    <tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Be aware that in TensorFlow all tensors are immutable, so in the latter case any changes in ``b`` cannot be reflected in the CuPy array ``a``.

Note that as of DLPack v0.4 for correctness it (implicitly) requires users to ensure that such conversion (both importing and exporting a CuPy array) must happen on the same CUDA/HIP stream. If in doubt, the current CuPy stream in use can be fetched by, for example, calling :func:`cupy.cuda.get_current_stream`. Please consult the other framework's documentation for how to access and control the streams. This requirement might be relaxed/changed in a future DLPack version.
