Difference between CuPy and NumPy
=================================

The interface of CuPy is designed to obey that of NumPy.
However, there are some differeneces.


Cast behavior from float to integer
-----------------------------------

Some casting behaviors from float to integer are not defined in C++ specification.
The casting from a negative float to unsigned integer and infinity to integer is one of such examples.
The behavior of NumPy depends on your CPU architecture.
This is Intel CPU result.

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

NumPy's random value generator does not support dtype option and it always resturns a ``float32`` value.
We support the option in CuPy because cuRAND, which is used in CuPy, supports any types of float values.

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


Reduction methods return zero-dimensional array
-----------------------------------------------

NumPy's reduction functions (e.g. :func:`numpy.sum`) return scalar values (e.g. :class:`numpy.float32`).
However CuPy counterparts return zero-dimensional :class:`cupy.ndarray` s.
That is because CuPy scalar values (e.g. :class:`cupy.float32`) are aliases of NumPy scalar values and are allocated in CPU memory.
If these types were returned, it would be required to synchronize between GPU and CPU.
If you want to use scalar values, cast the returned arrays explicitly.

  >>> type(np.sum(np.arange(3))) == np.int64
  True
  >>> type(cupy.sum(cupy.arange(3))) == cupy.core.core.ndarray
  True


Data types
----------

Data type of CuPy arrays cannot be non-numeric like strings and objects.


Array creation from Python objects
----------------------------------

Currently, an array cannot be created from Python object containing CuPy array.
For example, you cannot convert a list of CuPy arrays into CuPy array by :func:`cupy.array` or :func:`cupy.asarray`.
Use :func:`cupy.stack` instead.

  >>> data_cpu = [np.arange(10), np.arange(10)]
  >>> np.asarray(data_cpu)
  array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

  >>> data_gpu = [cupy.arange(10), cupy.arange(10)]
  >>> cupy.asarray(data_gpu)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ValueError: Unsupported dtype object
  >>> cupy.stack(data_gpu)
  array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])


Universal Functions only work with CuPy array or scalar
-------------------------------------------------------

Unlike NumPy, Universal Functions in CuPy only work with CuPy array or scalar.
They do not accept objects (e.g., lists or :class:`numpy.ndarray`).

  >>> np.power([np.arange(5)], 2)
  array([[ 0,  1,  4,  9, 16]])

  >>> cupy.power([cupy.arange(5)], 2)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: Unsupported type <class 'list'>
