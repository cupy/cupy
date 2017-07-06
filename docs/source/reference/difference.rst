Difference between CuPy and NumPy
=================================

The interface of CuPy is designed to obey that of NumPy.
However, there are some differeneces.


Cast behavior from float to integer
-----------------------------------

Some casting behaviors from float to integer are not defined in C++ specification.
The casting from a negative float to unsigned integer and infinity to integer is one of such eamples.
The behavior of NumPy depends on your CPU architecture.
This is Intel CPU result.

  >>> np.array([-1], dtype='f').astype('I')
  array([4294967295], dtype=uint32)
  >>> cupy.array([-1], dtype='f').astype('I')
  array([0], dtype=uint32)

  >>> np.array([float('inf')], dtype='f').astype('i')
  array([-2147483648], dtype=int32)
  >>> cupy.array([float('inf')], dtype='f').astype('i')
  array([2147483647], dtype=int32)


Boolean values squared
----------------------

In NumPy implementation, ``x ** 2`` is calculated using multiplication operator as ``x * x``.
Because the result of the multiplication of boolean values is boolean, ``True ** 2`` return boolean value.
However, when you use power operator with other arguments, it returns int values.
If we aligned the behavior of the squared boolean values of CuPy to that of NumPy, we would have to check their values in advance of the calculation.
But it would be slow because it would force CPUs to wait until the calculation on GPUs end.
So we decided not to check its value.

  >>> np.array([True]) ** 2
  array([ True], dtype=bool)
  >>> cupy.array([True]) ** 2
  array([1])


Random methods support dtype argument
-------------------------------------

NumPy's random value generator does not support dtype option and it always resturns a ``float32`` value.
We support the option in CuPy because cuRAND, which is used in CuPy, supports any types of float values.

  >>> np.random.randn(dtype='f')
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: randn() got an unexpected keyword argument 'dtype'
  >>> cupy.random.randn(dtype='f')    # doctest: +SKIP
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
  >>> v = cupy.arange(10000).astype(np.float)
  >>> a[i] = v
  >>> a  # doctest: +SKIP
  array([ 9150.,  9151.])

NumPy stores the value corresponding to the
last element among elements referencing duplicate locations.

  >>> a_cpu = np.zeros((2,))
  >>> i_cpu = np.arange(10000) % 2
  >>> v_cpu = np.arange(10000).astype(np.float)
  >>> a_cpu[i_cpu] = v_cpu
  >>> a_cpu
  array([ 9998.,  9999.])


Reduction methods return zero-dimensional array
-----------------------------------------------

NumPy's reduction methods such as :func:`numpy.sum` returns a scalar value such as :class:`numpy.float32`.
However CuPy's one returns zero-dimensional array because CuPy's scalar value such as :class:`cupy.float32` is allocated in CPU memory.
To return a scalar value it is required to synchronize GPU and CPU.
If you want to use a scalar value, cast a result array explicitly.

  >>> type(np.sum(np.arange(3)))
  <class 'numpy.int64'>
  >>> type(cupy.sum(cupy.arange(3)))
  <class 'cupy.core.core.ndarray'>
