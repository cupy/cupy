import unittest

import numpy
import six

import cupy
from cupy import testing


@testing.gpu
class TestSumprod(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_sum_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum(axis=1)

    @testing.slow
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_allclose()
    def test_sum_axis_huge(self, xp):
        a = testing.shaped_random((2048, 1, 1024), xp, 'b')
        a = xp.broadcast_to(a, (2048, 1024, 1024))
        return a.sum(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a, axis=1)

    # float16 is omitted, since NumPy's sum on float16 arrays has more error
    # than CuPy's.
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_sum_axis2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return a.sum(axis=1)

    def test_sum_axis2_float16(self):
        # Note that the above test example overflows in float16. We use a
        # smaller array instead.
        a = testing.shaped_arange((2, 30, 4), dtype='e')
        sa = a.sum(axis=1)
        b = testing.shaped_arange((2, 30, 4), numpy, dtype='f')
        sb = b.sum(axis=1)
        testing.assert_allclose(sa, sb.astype('e'))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axis_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return a.sum(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axis_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return a.sum(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.sum(axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_sum_axes2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return a.sum(axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_sum_axes3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.sum(axis=(0, 2, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_sum_axes4(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return a.sum(axis=(0, 2, 3))

    @testing.numpy_cupy_allclose()
    def test_sum_keepdims(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.sum(axis=1, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 4), dtype=dtype)
        a.sum(axis=1, out=b)
        return b

    def test_sum_out_wrong_shape(self):
        a = testing.shaped_arange((2, 3, 4))
        b = cupy.empty((2, 3))
        with self.assertRaises(ValueError):
            a.sum(axis=1, out=b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_prod_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.prod()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_prod_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.prod(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_prod_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.prod(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_prod_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.prod(a, axis=1)


axes = [0, 1, 2]


@testing.parameterize(*testing.product({'axis': axes}))
@testing.gpu
class TestCumsum(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.cumsum(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_2dim(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_axis(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(six.moves.range(4, 4 + n)), xp, dtype)
        return xp.cumsum(a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_raises()
    def test_invalid_axis_lower(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_raises()
    def test_invalid_axis_upper(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumsum(a, axis=a.ndim + 1)


@testing.gpu
class TestCumprod(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.cumprod(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_cumprod_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumprod(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumprod(a, axis=1)

    @testing.slow
    @testing.numpy_cupy_allclose()
    def test_cumprod_huge_array(self, xp):
        a = xp.ones(2 ** 32, 'b')
        return xp.cumprod(a)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_raises()
    def test_invalid_axis_lower(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumprod(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    @testing.with_requires('numpy>=1.13')
    @testing.numpy_cupy_raises()
    def test_invalid_axis_upper(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.cumprod(a, axis=a.ndim)
