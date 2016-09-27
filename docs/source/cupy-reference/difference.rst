Difference between CuPy and NumPy
=================================

The interface of CuPy is designed to obey the interface of NumPy.
However it has some differenece.


Cast behavior from float to integer
-----------------------------------

Some casting behavior from float to integer is not defined.
For example from a negative float to unsigned integer and infinity to integer.
It depends on your CPU.
This is Intel CPU result::

  >>> numpy.array([-1], dtype='f').astype('I')
  array([4294967295], dtype=uint32)
  >>> cupy.array([-1], dtype='f').astype('I')
  array([0], dtype=uint32)

  >>> numpy.array([float('inf')], dtype='f').astype('i')
  array([-2147483648], dtype=int32)
  >>> cupy.array([float('inf')], dtype='f').astype('i')
  array([2147483647], dtype=int32)

Note that NumPy itself behaves diffenently with differenct CPUs.


Boolean values squared
----------------------

In NumPy implementation, ``x ** 2`` is calculated using multiplication operator as ``x * x``.
Because result of multiplication of boolean values is boolean, ``True ** 2`` return boolean value.
However, when you use power operator with other arguments, it returns int values.
In CuPy, a CPU requires to wait GPU in order to check its value, and it is slow.
So we decided not to check its value::

  >>> numpy.array([True]) ** 2
  array([ True], dtype=bool)
  >>> cupy.array([True]) ** 2
  array([1])


Random methods support dtype argument
-------------------------------------

NumPy's random value generator does not support dtype option and it always resturns a ``float32`` value.
We support the option in CuPy because cuRAND which is used in CuPy support any types of float values::

  >>> numpy.random.randn(dtype='f')
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: randn() got an unexpected keyword argument 'dtype'
  >>> cupy.random.randn(dtype='f')
  array(0.10689262300729752, dtype=float32)
