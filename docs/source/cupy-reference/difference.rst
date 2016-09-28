Difference between CuPy and NumPy
=================================

The interface of CuPy is designed to obey that of NumPy.
However, there are some differeneces.


Cast behavior from float to integer
-----------------------------------

Some casting behaviors from float to integer are not defined in C++ specification.
The casting from a negative float to unsigned integer and infinity to integer is one of such eamples.
The behavior of NumPy depends on your CPU architecture.
This is Intel CPU result::

  >>> numpy.array([-1], dtype='f').astype('I')
  array([4294967295], dtype=uint32)
  >>> cupy.array([-1], dtype='f').astype('I')
  array([0], dtype=uint32)

  >>> numpy.array([float('inf')], dtype='f').astype('i')
  array([-2147483648], dtype=int32)
  >>> cupy.array([float('inf')], dtype='f').astype('i')
  array([2147483647], dtype=int32)


Boolean values squared
----------------------

In NumPy implementation, ``x ** 2`` is calculated using multiplication operator as ``x * x``.
Because the result of the multiplication of boolean values is boolean, ``True ** 2`` return boolean value.
However, when you use power operator with other arguments, it returns int values.
If we aligned the behavior of the squared boolean values of CuPy to that of NumPy, we would have to check their values in advance of the calculation.
But it would be slow because it would force CPUs to wait until the calculation on GPUs.
So we decided not to check its value::

  >>> numpy.array([True]) ** 2
  array([ True], dtype=bool)
  >>> cupy.array([True]) ** 2
  array([1])


Random methods support dtype argument
-------------------------------------

NumPy's random value generator does not support dtype option and it always resturns a ``float32`` value.
We support the option in CuPy because cuRAND, which is used in CuPy, supports any types of float values::

  >>> numpy.random.randn(dtype='f')
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: randn() got an unexpected keyword argument 'dtype'
  >>> cupy.random.randn(dtype='f')
  array(0.10689262300729752, dtype=float32)
