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


.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.testing.assert_allclose
   cupy.testing.assert_array_almost_equal
   cupy.testing.assert_array_almost_equal_nulp
   cupy.testing.assert_array_max_ulp
   cupy.testing.assert_array_equal
   cupy.testing.assert_array_list_equal
   cupy.testing.assert_array_less


NumPy-CuPy Consistency Check
----------------------------

The following decorators are for testing consistency
between CuPy's functions and corresponding NumPy's ones.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.testing.numpy_cupy_allclose
   cupy.testing.numpy_cupy_array_almost_equal
   cupy.testing.numpy_cupy_array_almost_equal_nulp
   cupy.testing.numpy_cupy_array_max_ulp
   cupy.testing.numpy_cupy_array_equal
   cupy.testing.numpy_cupy_array_list_equal
   cupy.testing.numpy_cupy_array_less
   cupy.testing.numpy_cupy_raises


Parameterized dtype Test
------------------------

The following decorators offer the standard way for
parameterized test with respect to single or the
combination of dtype(s).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.testing.for_dtypes
   cupy.testing.for_all_dtypes
   cupy.testing.for_float_dtypes
   cupy.testing.for_signed_dtypes
   cupy.testing.for_unsigned_dtypes
   cupy.testing.for_int_dtypes
   cupy.testing.for_dtypes_combination
   cupy.testing.for_all_dtypes_combination
   cupy.testing.for_signed_dtypes_combination
   cupy.testing.for_unsigned_dtypes_combination
   cupy.testing.for_int_dtypes_combination


Parameterized order Test
------------------------
The following decorators offer the standard way to parameterize tests with
orders.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cupy.testing.for_orders
   cupy.testing.for_CF_orders
