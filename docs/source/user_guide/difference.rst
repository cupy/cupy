Differences between CuPy and NumPy
==================================

The interface of CuPy is designed to obey that of NumPy.
However, there are some differences.


Cast behavior from float to integer
-----------------------------------

Some casting behaviors from float to integer are not defined in C++ specification.
The casting from a negative float to unsigned integer and infinity to integer is one of such examples.
The behavior of NumPy depends on your CPU architecture.
This is the result on an Intel CPU:

  >>> np.array([-1], dtype=np.float32).astype(np.uint32)
  array([4294967295], dtype=uint32)
  >>> cupy.array([-1], dtype=np.float32).astype(np.uint32)
  array([0], dtype=uint32)

  >>> np.array([float('inf')], dtype=np.float32).astype(np.int32)
  array([-2147483648], dtype=int32)
  >>> cupy.array([float('inf')], dtype=np.float32).astype(np.int32)
  array([2147483647], dtype=int32)


Random methods support dtype argument
-------------------------------------

NumPy's random value generator does not support a `dtype` argument and instead always returns a ``float64`` value.
We support the option in CuPy because cuRAND, which is used in CuPy, supports both ``float32`` and ``float64``.


  >>> np.random.randn(dtype=np.float32)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: randn() got an unexpected keyword argument 'dtype'
  >>> cupy.random.randn(dtype=np.float32)    # doctest: +SKIP
  array(0.10689262300729752, dtype=float32)


Out-of-bounds indices
---------------------
CuPy handles out-of-bounds indices differently by default from NumPy when
using integer array indexing.
NumPy handles them by raising an error, but CuPy wraps around them.

  >>> x = np.array([0, 1, 2])
  >>> x[[1, 3]] = 10
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  IndexError: index 3 is out of bounds for axis 1 with size 3
  >>> x = cupy.array([0, 1, 2])
  >>> x[[1, 3]] = 10
  >>> x
  array([10, 10,  2])


Duplicate values in indices
---------------------------
CuPy's ``__setitem__`` behaves differently from NumPy when integer arrays
reference the same location multiple times.
In that case, the value that is actually stored is undefined.
Here is an example of CuPy.

  >>> a = cupy.zeros((2,))
  >>> i = cupy.arange(10000) % 2
  >>> v = cupy.arange(10000).astype(np.float32)
  >>> a[i] = v
  >>> a  # doctest: +SKIP
  array([ 9150.,  9151.])

NumPy stores the value corresponding to the
last element among elements referencing duplicate locations.

  >>> a_cpu = np.zeros((2,))
  >>> i_cpu = np.arange(10000) % 2
  >>> v_cpu = np.arange(10000).astype(np.float32)
  >>> a_cpu[i_cpu] = v_cpu
  >>> a_cpu
  array([9998., 9999.])


Zero-dimensional array
-----------------------------------------------

Reduction methods
~~~~~~~~~~~~~~~~~

NumPy's reduction functions (e.g. :func:`numpy.sum`) return scalar values (e.g. :class:`numpy.float32`).
However CuPy counterparts return zero-dimensional :class:`cupy.ndarray` s.
That is because CuPy scalar values (e.g. :class:`cupy.float32`) are aliases of NumPy scalar values and are allocated in CPU memory.
If these types were returned, it would be required to synchronize between GPU and CPU.
If you want to use scalar values, cast the returned arrays explicitly.

  >>> type(np.sum(np.arange(3))) == np.int64
  True
  >>> type(cupy.sum(cupy.arange(3))) == cupy.ndarray
  True


Type promotion
~~~~~~~~~~~~~~

CuPy automatically promotes dtypes of :class:`cupy.ndarray` s in a function with two or more operands, the result dtype is determined by the dtypes of the inputs.
This is different from NumPy's rule on type promotion, when operands contain zero-dimensional arrays.
Zero-dimensional :class:`numpy.ndarray` s are treated as if they were scalar values if they appear in operands of NumPy's function,
This may affect the dtype of its output, depending on the values of the "scalar" inputs.

  >>> (np.array(3, dtype=np.int32) * np.array([1., 2.], dtype=np.float32)).dtype
  dtype('float32')
  >>> (np.array(300000, dtype=np.int32) * np.array([1., 2.], dtype=np.float32)).dtype
  dtype('float64')
  >>> (cupy.array(3, dtype=np.int32) * cupy.array([1., 2.], dtype=np.float32)).dtype
  dtype('float64')


Matrix type (:class:`numpy.matrix`)
-----------------------------------

SciPy returns :class:`numpy.matrix` (a subclass of :class:`numpy.ndarray`) when dense matrices are computed from sparse matrices (e.g., ``coo_matrix + ndarray``). However, CuPy returns :class:`cupy.ndarray` for such operations.

There is no plan to provide :class:`numpy.matrix` equivalent in CuPy.
This is because the use of :class:`numpy.matrix` is no longer recommended since NumPy 1.15.


Data types
----------

Data type of CuPy arrays cannot be non-numeric like strings or objects.
See :ref:`overview` for details.


Universal Functions only work with CuPy array or scalar
-------------------------------------------------------

Unlike NumPy, Universal Functions in CuPy only work with CuPy array or scalar.
They do not accept other objects (e.g., lists or :class:`numpy.ndarray`).

  >>> np.power([np.arange(5)], 2)
  array([[ 0,  1,  4,  9, 16]])

  >>> cupy.power([cupy.arange(5)], 2)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: Unsupported type <class 'list'>


Random seed arrays are hashed to scalars
----------------------------------------

Like Numpy, CuPy's RandomState objects accept seeds either as numbers or as
full numpy arrays.

  >>> seed = np.array([1, 2, 3, 4, 5])
  >>> rs = cupy.random.RandomState(seed=seed)

However, unlike Numpy, array seeds will be hashed down to a single number and
so may not communicate as much entropy to the underlying random number
generator.


NaN (not-a-number) handling
---------------------------

By default CuPy's reduction functions (e.g., :func:`cupy.sum`) handle NaNs in complex numbers differently from NumPy's
counterparts:

  >>> a = [0.5 + 3.7j, complex(0.7, np.nan), complex(np.nan, -3.9), complex(np.nan, np.nan)]
  >>>
  >>> a_np = np.asarray(a)
  >>> print(a_np.max(), a_np.min())
  (0.7+nanj) (0.7+nanj)
  >>>
  >>> a_cp = cp.asarray(a_np)
  >>> print(a_cp.max(), a_cp.min())
  (nan-3.9j) (nan-3.9j)

The reason is that internally the reduction is performed in a strided fashion, thus it does not ensure a proper
comparison order and cannot follow NumPy's rule to always propagate the first-encountered NaN.
Note that this difference does not apply when CUB is enabled (which is the default for CuPy v11 or later.)

Contiguity / Strides
--------------------

To provide the best performance, the contiguity of a resulting ndarray is not guaranteed to match with that of NumPy's output.

  >>> a = np.array([[1, 2], [3, 4]], order='F')
  >>> print((a + a).flags.f_contiguous)
  True

  >>> a = cp.array([[1, 2], [3, 4]], order='F')
  >>> print((a + a).flags.f_contiguous)
  False
