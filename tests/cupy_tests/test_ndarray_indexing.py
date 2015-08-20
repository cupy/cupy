import unittest

from cupy import testing


@testing.gpu
class TestArrayIndexing(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_getitem(self, shape, transpose, indexes, xp, dtype):
        a = testing.shaped_arange(shape, xp, dtype)
        if transpose:
            a = a.transpose(transpose)
        return a[indexes]

    def test_getitem_direct(self):
        self.check_getitem((2, 3, 4), None, (1, 0, 2))

    def test_getitem_transposed_direct(self):
        self.check_getitem((2, 3, 4), (2, 0, 1), (1, 0, 2))

    def test_getitem_slice1(self):
        self.check_getitem((2, 3, 4), None,
                           (slice(None), slice(None, 1), slice(2)))

    def test_getitem_transposed_slice1(self):
        self.check_getitem((2, 3, 4), (2, 0, 1),
                           (slice(None), slice(None, 1), slice(2)))

    def test_getitem_slice2(self):
        self.check_getitem((2, 3, 5), None,
                           (slice(None, None, -1), slice(1, None, -1),
                            slice(4, 1, -2)))

    def test_getitem_transposed_slice2(self):
        self.check_getitem((2, 3, 5), (2, 0, 1),
                           (slice(4, 1, -2), slice(None, None, -1),
                            slice(1, None, -1)))

    def test_getitem_ellipsis1(self):
        self.check_getitem((2, 3, 4), None, (Ellipsis, 2))

    def test_getitem_ellipsis2(self):
        self.check_getitem((2, 3, 4), None, (1, Ellipsis))

    def test_getitem_ellipsis3(self):
        self.check_getitem((2, 3, 4, 5), None, (1, Ellipsis, 3))

    def test_getitem_newaxis1(self):
        self.check_getitem((2, 3, 4), None, (1, None, slice(2), None, 2))

    def test_getitem_newaxis2(self):
        self.check_getitem((2, 3), None, (None,))

    def test_getitem_newaxis3(self):
        self.check_getitem((2,), None, (slice(None,), None))

    def test_getitem_newaxis4(self):
        self.check_getitem((), None, (None,))

    def test_getitem_newaxis5(self):
        self.check_getitem((), None, (None, None))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_constant(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        a[:] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_partial_constant(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        a[1, 1:3] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_copy(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = testing.shaped_arange((2, 3, 4), xp, dtype)
        a[:] = b
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_partial_copy(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = testing.shaped_arange((3, 2), xp, dtype)
        a[1, ::-1, 1:4:2] = b
        return a

    @testing.numpy_cupy_array_equal()
    def test_T(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.T

    @testing.numpy_cupy_array_equal()
    def test_T_vector(self, xp):
        a = testing.shaped_arange((4,), xp)
        return a.T
