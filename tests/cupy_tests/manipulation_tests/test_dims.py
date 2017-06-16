import unittest

import cupy
from cupy import testing


@testing.gpu
class TestDims(unittest.TestCase):

    _multiprocess_can_split_ = True

    def check_atleast(self, func, xp):
        a = testing.shaped_arange((), xp)
        b = testing.shaped_arange((2,), xp)
        c = testing.shaped_arange((2, 2), xp)
        d = testing.shaped_arange((4, 3, 2), xp)
        return func(a, b, c, d)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_1d1(self, xp):
        return self.check_atleast(xp.atleast_1d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_1d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_1d(a)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_2d1(self, xp):
        return self.check_atleast(xp.atleast_2d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_2d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_2d(a)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_3d1(self, xp):
        return self.check_atleast(xp.atleast_3d, xp)

    @testing.numpy_cupy_array_equal()
    def test_atleast_3d2(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        return xp.atleast_3d(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_arrays(self, xp, dtype):
        a = testing.shaped_arange((2, 1, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 1, 4), xp, dtype)
        c, d = xp.broadcast_arrays(a, b)
        return d

    def test_broadcast(self):
        a = testing.shaped_arange((2, 1, 3, 4))
        b = testing.shaped_arange((3, 1, 4))
        bc = cupy.broadcast(a, b)
        self.assertEqual((2, 3, 3, 4), bc.shape)
        self.assertEqual(2 * 3 * 3 * 4, bc.size)
        self.assertEqual(4, bc.nd)

    @testing.numpy_cupy_raises()
    def test_broadcast_fail(self, xp):
        a = xp.zeros((2, 3))
        b = xp.zeros((3, 2))
        xp.broadcast(a, b)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_to(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        b = xp.broadcast_to(a, (2, 3, 3, 4))
        return b

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_broadcast_to_fail(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        xp.broadcast_to(a, (1, 3, 4))

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_broadcast_to_short_shape(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((1, 3, 4), xp, dtype)
        xp.broadcast_to(a, (3, 4))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_to_numpy19(self, xp, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), xp, dtype)
        if xp is cupy:
            b = xp.broadcast_to(a, (2, 3, 3, 4))
        else:
            dummy = xp.empty((2, 3, 3, 4))
            b, _ = xp.broadcast_arrays(a, dummy)
        return b

    @testing.for_all_dtypes()
    def test_broadcast_to_fail_numpy19(self, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((3, 1, 4), cupy, dtype)
        with self.assertRaises(ValueError):
            cupy.broadcast_to(a, (1, 3, 4))

    @testing.for_all_dtypes()
    def test_broadcast_to_short_shape_numpy19(self, dtype):
        # Note that broadcast_to is only supported on numpy>=1.10
        a = testing.shaped_arange((1, 3, 4), cupy, dtype)
        with self.assertRaises(ValueError):
            cupy.broadcast_to(a, (3, 4))

    @testing.numpy_cupy_array_equal()
    def test_expand_dims0(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 0)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 1)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, 2)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims_negative1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, -2)

    @testing.numpy_cupy_array_equal()
    def test_expand_dims_negative2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.expand_dims(a, -4)

    @testing.numpy_cupy_array_equal()
    def test_squeeze1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze()

    @testing.numpy_cupy_array_equal()
    def test_squeeze2(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.squeeze()

    @testing.numpy_cupy_array_equal()
    def test_squeze_int_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=2)

    @testing.numpy_cupy_array_equal()
    def test_squeze_int_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=-3)

    @testing.numpy_cupy_raises()
    def test_squeze_int_axis_failure(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        a.squeeze(axis=-9)

    @testing.numpy_cupy_array_equal()
    def test_squeze_tuple_axis1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(2, 4))

    @testing.numpy_cupy_array_equal()
    def test_squeze_tuple_axis2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(-4, -3))

    @testing.numpy_cupy_array_equal()
    def test_squeze_tuple_axis3(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=(4, 2))

    @testing.numpy_cupy_array_equal()
    def test_squeze_tuple_axis4(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return a.squeeze(axis=())

    @testing.numpy_cupy_raises()
    def test_squeze_tuple_axis_failure1(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        a.squeeze(axis=(-9,))

    @testing.numpy_cupy_raises()
    def test_squeze_tuple_axis_failure2(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        a.squeeze(axis=(2, 2))

    @testing.numpy_cupy_array_equal()
    def test_squeeze_scalar1(self, xp):
        a = testing.shaped_arange((), xp)
        return a.squeeze(axis=0)

    @testing.numpy_cupy_array_equal()
    def test_squeeze_scalar2(self, xp):
        a = testing.shaped_arange((), xp)
        return a.squeeze(axis=-1)

    @testing.numpy_cupy_raises()
    def test_squeeze_scalar_failure1(self, xp):
        a = testing.shaped_arange((), xp)
        a.squeeze(axis=-2)

    @testing.numpy_cupy_raises()
    def test_squeeze_scalar_failure2(self, xp):
        a = testing.shaped_arange((), xp)
        a.squeeze(axis=1)

    @testing.numpy_cupy_raises()
    def test_squeeze_failure(self, xp):
        a = testing.shaped_arange((2, 1, 3, 4), xp)
        a.squeeze(axis=2)

    @testing.numpy_cupy_array_equal()
    def test_external_squeeze(self, xp):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xp)
        return xp.squeeze(a)
