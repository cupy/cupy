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
