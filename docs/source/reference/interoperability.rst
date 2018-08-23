Interoperability
================

CuPy can also be used in conjuction with other frameworks.

NumPy
-----

:class:`cupy.ndarray` implements ``__array_ufunc__`` interface (see `NEP 13 — A Mechanism for Overriding Ufuncs <http://www.numpy.org/neps/nep-0013-ufunc-overrides.html>`_ for details).
This enables NumPy ufuncs to be directly operated on CuPy arrays.

.. code:: python

    import cupy
    import numpy

    arr = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)
    result = numpy.sum(arr)
    print(type(result))  # <class 'cupy.core.core.ndarray'>

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
	cupy_array = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)

	# Convert it into a DLPack tensor.
	dlpack_array = cupy_array.toDlpack()

	# Convert it back to a CuPy array.
	cupy_array2 = cupy.fromDlpack(dlpack_array)

Here is an example of converting PyTorch tensor into :class:`cupy.ndarray`.

.. code:: python

	import cupy
	import torch

	from torch.utils.dlpack import to_dlpack
	from torch.utils.dlpack import from_dlpack

	# Create a PyTorch tensor.
	tx = torch.randn(1, 2, 3, 4).cuda()

	# Convert it into a DLPack tensor.
	t1 = to_dlpack(tx)

	# Convert it into a CuPy array.
	cx = cupy.fromDlpack(t1)

	# Convert it back to a PyTorch tensor.
	t2 = from_dlpack(cx.toDlpack())
