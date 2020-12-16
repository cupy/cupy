import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(2, 3), (), (4,)],
}))
@testing.gpu
class TestShape(unittest.TestCase):

    def test_shape(self):
        shape = self.shape
        for xp in (numpy, cupy):
            a = testing.shaped_arange(shape, xp)
            assert cupy.shape(a) == shape

    def test_shape_list(self):
        shape = self.shape
        a = testing.shaped_arange(shape, numpy)
        a = a.tolist()
        assert cupy.shape(a) == shape


@testing.gpu
class TestReshape(unittest.TestCase):

    def test_reshape_strides(self):
        def func(xp):
            a = testing.shaped_arange((1, 1, 1, 2, 2), xp)
            return a.strides
        assert func(numpy) == func(cupy)

    def test_reshape2(self):
        def func(xp):
            a = xp.zeros((8,), dtype=xp.float32)
            return a.reshape((1, 1, 1, 4, 1, 2)).strides
        assert func(numpy) == func(cupy)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nocopy_reshape(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2, order=order)
        b[1] = 1
        return a

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nocopy_reshape_with_order(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2, order=order)
        b[1] = 1
        return a

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_transposed_reshape2(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.reshape(2, 3, 4, order=order)

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_reshape_with_unknown_dimension(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.reshape(3, -1, order=order)

    def test_reshape_with_multiple_unknown_dimensions(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                a.reshape(3, -1, -1)

    def test_reshape_with_changed_arraysize(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                a.reshape(2, 4, 4)

    def test_reshape_invalid_order(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                a.reshape(2, 4, 4, order='K')

    def test_reshape_zerosize_invalid(self):
        for xp in (numpy, cupy):
            a = xp.zeros((0,))
            with pytest.raises(ValueError):
                a.reshape(())

    @testing.numpy_cupy_array_equal()
    def test_reshape_zerosize(self, xp):
        a = xp.zeros((0,))
        return a.reshape((0,))

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_external_reshape(self, xp, order):
        a = xp.zeros((8,), dtype=xp.float32)
        return xp.reshape(a, (1, 1, 1, 4, 1, 2), order=order)


@testing.gpu
class TestRavel(unittest.TestCase):

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_ravel(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return a.ravel(order)

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_ravel2(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.ravel(order)

    @testing.for_orders('CFA')
    @testing.numpy_cupy_array_equal()
    def test_ravel3(self, xp, order):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = xp.asfortranarray(a)
        return a.ravel(order)

    @testing.numpy_cupy_array_equal()
    def test_external_ravel(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        a = a.transpose(2, 0, 1)
        return xp.ravel(a)


@testing.parameterize(*testing.product({
    'order_init': ['C', 'F'],
    'order_reshape': ['C', 'F', 'A', 'c', 'f', 'a'],
    'shape_in_out': [((2, 3), (1, 6, 1)),  # (shape_init, shape_final)
                     ((6,), (2, 3)),
                     ((3, 3, 3), (9, 3))],
}))
@testing.gpu
class TestReshapeOrder(unittest.TestCase):

    def test_reshape_contiguity(self):
        shape_init, shape_final = self.shape_in_out

        a_cupy = testing.shaped_arange(shape_init, xp=cupy)
        a_cupy = cupy.asarray(a_cupy, order=self.order_init)
        b_cupy = a_cupy.reshape(shape_final, order=self.order_reshape)

        a_numpy = testing.shaped_arange(shape_init, xp=numpy)
        a_numpy = numpy.asarray(a_numpy, order=self.order_init)
        b_numpy = a_numpy.reshape(shape_final, order=self.order_reshape)

        assert b_cupy.flags.f_contiguous == b_numpy.flags.f_contiguous
        assert b_cupy.flags.c_contiguous == b_numpy.flags.c_contiguous

        testing.assert_array_equal(b_cupy.strides, b_numpy.strides)
        testing.assert_array_equal(b_cupy, b_numpy)
