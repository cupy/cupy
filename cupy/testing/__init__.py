from cupy.testing import array
from cupy.testing import attr
from cupy.testing import helper
from cupy.testing import parameterized

assert_allclose = array.assert_allclose
assert_array_almost_equal = array.assert_array_almost_equal
assert_array_almost_equal_nulp = array.assert_array_almost_equal_nulp
assert_array_max_ulp = array.assert_array_max_ulp
assert_array_equal = array.assert_array_equal
assert_array_list_equal = array.assert_array_list_equal
assert_array_less = array.assert_array_less

numpy_cupy_allclose = helper.numpy_cupy_allclose
numpy_cupy_array_almost_equal = helper.numpy_cupy_array_almost_equal
numpy_cupy_array_almost_equal_nulp = \
    helper.numpy_cupy_array_almost_equal_nulp
numpy_cupy_array_max_ulp = helper.numpy_cupy_array_max_ulp
numpy_cupy_array_equal = helper.numpy_cupy_array_equal
numpy_cupy_array_list_equal = helper.numpy_cupy_array_list_equal
numpy_cupy_array_less = helper.numpy_cupy_array_less
numpy_cupy_raises = helper.numpy_cupy_raises
for_dtypes = helper.for_dtypes
for_all_dtypes = helper.for_all_dtypes
for_float_dtypes = helper.for_float_dtypes
for_signed_dtypes = helper.for_signed_dtypes
for_unsigned_dtypes = helper.for_unsigned_dtypes
for_int_dtypes = helper.for_int_dtypes
for_dtypes_combination = helper.for_dtypes_combination
for_all_dtypes_combination = helper.for_all_dtypes_combination
for_signed_dtypes_combination = helper.for_signed_dtypes_combination
for_unsigned_dtypes_combination = helper.for_unsigned_dtypes_combination
for_int_dtypes_combination = helper.for_int_dtypes_combination

with_requires = helper.with_requires

parameterize = parameterized.parameterize
product = parameterized.product

shaped_arange = helper.shaped_arange
shaped_reverse_arange = helper.shaped_reverse_arange

shaped_random = helper.shaped_random

NumpyError = helper.NumpyError

gpu = attr.gpu
multi_gpu = attr.multi_gpu
