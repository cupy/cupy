Testing Modules
===============

.. module:: cupy.testing

CuPy offers testing utilities to support unit testing.
They are under namespace :mod:`cupy.testing`.


Standard Assertions
-------------------

The assertions have same names as NumPy's ones.
The difference from NumPy is that they can accept both :class:`numpy.ndarray`
and :class:`cupy.ndarray`.


.. autofunction:: assert_allclose
.. autofunction:: assert_array_almost_equal
.. autofunction:: assert_array_almost_equal_nulp
.. autofunction:: assert_array_max_ulp
.. autofunction:: assert_array_equal
.. autofunction:: assert_array_list_equal
.. autofunction:: assert_array_less


NumPy-CuPy Consistency Check
----------------------------

The following decorators are for testing consistency
between CuPy's functions and corresponding NumPy's ones.

.. autofunction:: numpy_cupy_allclose
.. autofunction:: numpy_cupy_array_almost_equal
.. autofunction:: numpy_cupy_array_almost_equal_nulp
.. autofunction:: numpy_cupy_array_max_ulp
.. autofunction:: numpy_cupy_array_equal
.. autofunction:: numpy_cupy_array_list_equal
.. autofunction:: numpy_cupy_array_less
.. autofunction:: numpy_cupy_raises


Parameterized dtype Test
------------------------

The following decorators offers the standard way for
parameterized test with respect to single or the
combination of dtype(s).

.. autofunction:: for_dtypes
.. autofunction:: for_all_dtypes
.. autofunction:: for_float_dtypes
.. autofunction:: for_signed_dtypes
.. autofunction:: for_unsigned_dtypes
.. autofunction:: for_int_dtypes
.. autofunction:: for_dtypes_combination
.. autofunction:: for_all_dtypes_combination
.. autofunction:: for_signed_dtypes_combination
.. autofunction:: for_unsigned_dtypes_combination
.. autofunction:: for_int_dtypes_combination
