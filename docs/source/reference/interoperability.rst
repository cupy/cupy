Interoperability
================

CuPy can also be used in conjuction with other frameworks.

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
    print(type(result))  # => <class 'cupy.core.core.ndarray'>

:class:`cupy.ndarray` also implements ``__array_function__`` interface (see `NEP 18 — A dispatch mechanism for NumPy’s high level array functions <http://www.numpy.org/neps/nep-0018-array-function-protocol.html>`_ for details).
This enables code using NumPy to be directly operated on CuPy arrays.
``__array_function__`` feature requires NumPy 1.16 or later; note that this is currently defined as an experimental feature of NumPy and you need to specify the environment variable (``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1``) to enable it.

Numba
-----

`Numba <https://numba.pydata.org/>`_ is a Python JIT compiler with NumPy support.

:class:`cupy.ndarray` implements ``__cuda_array_interface__``, which is the CUDA array interchange interface compatible with Numba v0.39.0 or later (see `CUDA Array Interface <http://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html>`_ for details).
It means you can pass CuPy arrays to kernels JITed with Numba.
The folowing is a simple example code borrowed from `numba/numba#2860 <https://github.com/numba/numba/pull/2860>`_:

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

mpi4py
------

`MPI for Python (mpi4py) <https://mpi4py.readthedocs.io/en/latest/>`_ is a Python wrapper for the Message Passing Interface (MPI) libraries.

MPI is the most widely used standard for high-performance inter-process communications. Recently several MPI vendors, including Open MPI and MVAPICH, have extended their support beyond the v3.1 standard to enable "CUDA-awareness"; that is, passing CUDA device pointers directly to MPI calls to avoid explicit data movement between the host and the device.

With the aforementioned ``__cuda_array_interface__`` standard implemented in CuPy, mpi4py now provides (experimental) support for passing CuPy arrays to MPI calls, provided that mpi4py is built against a CUDA-aware MPI implementation. The folowing is a simple example code borrowed from `mpi4py Tutorial <https://mpi4py.readthedocs.io/en/latest/tutorial.html>`_:

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

DLPack
------

`DLPack <https://github.com/dmlc/dlpack>`_ is a specification of tensor structure to share tensors among frameworks.

CuPy supports importing from and exporting to DLPack data structure (:func:`cupy.fromDlpack` and :func:`cupy.ndarray.toDlpack`).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.fromDlpack

Here is a simple example:

.. code:: python

	import cupy

	# Create a CuPy array.
	cx1 = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)

	# Convert it into a DLPack tensor.
	dx = cx1.toDlpack()

	# Convert it back to a CuPy array.
	cx2 = cupy.fromDlpack(dx)

Here is an example of converting PyTorch tensor into :class:`cupy.ndarray`.

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
