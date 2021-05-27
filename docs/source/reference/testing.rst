.. module:: cupy.testing

Test support (:mod:`cupy.testing`)
==================================

.. Hint:: `NumPy API Reference: Test support (numpy.testing) <https://numpy.org/doc/stable/reference/routines.testing.html>`_

Asserts
-------

.. Hint:: These APIs can accept both :class:`numpy.ndarray` and :class:`cupy.ndarray`.

.. autosummary::
   :toctree: generated/

   assert_array_almost_equal
   assert_allclose
   assert_array_almost_equal_nulp
   assert_array_max_ulp
   assert_array_equal
   assert_array_less

CuPy-specific APIs
------------------

Asserts
~~~~~~~

.. autosummary::
   :toctree: generated/

   assert_array_list_equal


NumPy-CuPy Consistency Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following decorators are for testing consistency
between CuPy's functions and corresponding NumPy's ones.

.. autosummary::
   :toctree: generated/

   numpy_cupy_allclose
   numpy_cupy_array_almost_equal
   numpy_cupy_array_almost_equal_nulp
   numpy_cupy_array_max_ulp
   numpy_cupy_array_equal
   numpy_cupy_array_list_equal
   numpy_cupy_array_less


Parameterized dtype Test
~~~~~~~~~~~~~~~~~~~~~~~~

The following decorators offer the standard way for
parameterized test with respect to single or the
combination of dtype(s).

.. autosummary::
   :toctree: generated/

   for_dtypes
   for_all_dtypes
   for_float_dtypes
   for_signed_dtypes
   for_unsigned_dtypes
   for_int_dtypes
   for_complex_dtypes
   for_dtypes_combination
   for_all_dtypes_combination
   for_signed_dtypes_combination
   for_unsigned_dtypes_combination
   for_int_dtypes_combination


Parameterized order Test
~~~~~~~~~~~~~~~~~~~~~~~~

The following decorators offer the standard way to parameterize tests with
orders.

.. autosummary::
   :toctree: generated/

   for_orders
   for_CF_orders
