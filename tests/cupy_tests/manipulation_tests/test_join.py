import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy import cuda


@testing.gpu
class TestJoin(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_column_stack(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((4, 3), xp, dtype1)
        b = testing.shaped_arange((4,), xp, dtype2)
        c = testing.shaped_arange((4, 2), xp, dtype1)
        return xp.column_stack((a, b, c))

    def test_column_stack_wrong_ndim1(self):
        a = cupy.zeros(())
        b = cupy.zeros((3,))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_ndim2(self):
        a = cupy.zeros((3, 2, 3))
        b = cupy.zeros((3, 2))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_shape(self):
        a = cupy.zeros((3, 2))
        b = cupy.zeros((4, 3))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=2)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_axis_none(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((3, 5, 2), xp, dtype)
        c = testing.shaped_arange((7, ), xp, dtype)
        return xp.concatenate((a, b, c), axis=None)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xp, dtype)
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 5), xp, dtype)
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 1), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 1), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_4(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_5(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, 'i')
        return xp.concatenate((a, b) * 10, axis=-1)

    @testing.multi_gpu(2)
    def test_concatenate_large_different_devices(self):
        arrs = []
        for i in range(10):
            with cuda.Device(i % 2):
                arrs.append(cupy.empty((2, 3, 4)))
        with pytest.raises(ValueError):
            cupy.concatenate(arrs)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        return xp.concatenate((a, b, c), axis=-1)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_large_f_contiguous(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 3, 2), xp, dtype).T
        c = testing.shaped_arange((2, 3, 3), xp, dtype)
        d = testing.shaped_arange((2, 3, 2), xp, dtype).T
        e = testing.shaped_arange((2, 3, 2), xp, dtype)
        return xp.concatenate((a, b, c, d, e) * 2, axis=-1)

    @testing.numpy_cupy_array_equal()
    def test_concatenate_many_multi_dptye(self, xp):
        a = testing.shaped_arange((2, 1), xp, 'i')
        b = testing.shaped_arange((2, 1), xp, 'f')
        return xp.concatenate((a, b) * 1024, axis=1)

    @testing.slow
    def test_concatenate_32bit_boundary(self):
        a = cupy.zeros((2 ** 30,), dtype=cupy.int8)
        b = cupy.zeros((2 ** 30,), dtype=cupy.int8)
        ret = cupy.concatenate([a, b])
        del a
        del b
        del ret
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()

    def test_concatenate_wrong_ndim(self):
        a = cupy.empty((2, 3))
        b = cupy.empty((2,))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b))

    def test_concatenate_wrong_shape(self):
        a = cupy.empty((2, 3, 4))
        b = cupy.empty((3, 3, 4))
        c = cupy.empty((4, 4, 4))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b, c))

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate_out(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((3, 4), xp, dtype)
        c = testing.shaped_arange((3, 4), xp, dtype)
        out = xp.zeros((3, 12), dtype=dtype)
        xp.concatenate((a, b, c), axis=1, out=out)
        return out

    @testing.numpy_cupy_array_equal()
    def test_concatenate_out_same_kind(self, xp):
        a = testing.shaped_arange((3, 4), xp, xp.float64)
        b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
        c = testing.shaped_arange((3, 4), xp, xp.float64)
        out = xp.zeros((3, 12), dtype=xp.float32)
        xp.concatenate((a, b, c), axis=1, out=out)
        return out

    def test_concatenate_out_invalid_shape(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((4, 10), dtype=xp.float64)
            with pytest.raises(ValueError):
                xp.concatenate((a, b, c), axis=1, out=out)

    def test_concatenate_out_invalid_shape_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((2, 2, 10), dtype=xp.float64)
            with pytest.raises(ValueError):
                xp.concatenate((a, b, c), axis=1, out=out)

    def test_concatenate_out_invalid_dtype(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((3, 12), dtype=xp.int64)
            with pytest.raises(TypeError):
                xp.concatenate((a, b, c), axis=1, out=out)

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal()
    def test_concatenate_different_dtype(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((3, 4), xp, dtype1)
        b = testing.shaped_arange((3, 4), xp, dtype2)
        return xp.concatenate((a, b))

    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_array_equal(accept_error=TypeError)
    def test_concatenate_out_different_dtype(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((3, 4), xp, dtype1)
        b = testing.shaped_arange((3, 4), xp, dtype1)
        out = xp.zeros((6, 4), dtype=dtype2)
        return xp.concatenate((a, b), out=out)

    @testing.numpy_cupy_array_equal()
    def test_dstack(self, xp):
        a = testing.shaped_arange((1, 3, 2), xp)
        b = testing.shaped_arange((3,), xp)
        c = testing.shaped_arange((1, 3), xp)
        return xp.dstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element(self, xp):
        a = testing.shaped_arange((1, 2, 3), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element_2(self, xp):
        a = testing.shaped_arange((1, 2), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_dstack_single_element_3(self, xp):
        a = testing.shaped_arange((1,), xp)
        return xp.dstack((a,))

    @testing.numpy_cupy_array_equal()
    def test_hstack_vectors(self, xp):
        a = xp.arange(3)
        b = xp.arange(2, -1, -1)
        return xp.hstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_hstack_scalars(self, xp):
        a = testing.shaped_arange((), xp)
        b = testing.shaped_arange((), xp)
        c = testing.shaped_arange((), xp)
        return xp.hstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_hstack(self, xp):
        a = testing.shaped_arange((2, 1), xp)
        b = testing.shaped_arange((2, 2), xp)
        c = testing.shaped_arange((2, 3), xp)
        return xp.hstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_vstack_vectors(self, xp):
        a = xp.arange(3)
        b = xp.arange(2, -1, -1)
        return xp.vstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_vstack_single_element(self, xp):
        a = xp.arange(3)
        return xp.vstack((a,))

    def test_vstack_wrong_ndim(self):
        a = cupy.empty((3,))
        b = cupy.empty((3, 1))
        with self.assertRaises(ValueError):
            cupy.vstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_stack(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        b = testing.shaped_arange((2, 3), xp)
        c = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, b, c))

    def test_stack_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        b = testing.shaped_arange((2, 3), cupy)
        c = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, b, c))
        assert s.shape == (3, 2, 3)
        cupy.testing.assert_array_equal(s[0], a)
        cupy.testing.assert_array_equal(s[1], b)
        cupy.testing.assert_array_equal(s[2], c)

    @testing.numpy_cupy_array_equal()
    def test_stack_with_axis1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, a), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_stack_with_axis2(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, a), axis=2)

    def test_stack_with_axis_over(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp)
            with pytest.raises(ValueError):
                xp.stack((a, a), axis=3)

    def test_stack_with_axis_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, a), axis=1)

        assert s.shape == (2, 2, 3)
        cupy.testing.assert_array_equal(s[:, 0, :], a)
        cupy.testing.assert_array_equal(s[:, 1, :], a)

    @testing.numpy_cupy_array_equal()
    def test_stack_with_negative_axis(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.stack((a, a), axis=-1)

    def test_stack_with_negative_axis_value(self):
        a = testing.shaped_arange((2, 3), cupy)
        s = cupy.stack((a, a), axis=-1)

        assert s.shape == (2, 3, 2)
        cupy.testing.assert_array_equal(s[:, :, 0], a)
        cupy.testing.assert_array_equal(s[:, :, 1], a)

    def test_stack_different_shape(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp)
            b = testing.shaped_arange((2, 4), xp)
            with pytest.raises(ValueError):
                xp.stack([a, b])

    def test_stack_out_of_bounds1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp)
            with pytest.raises(ValueError):
                xp.stack([a, a], axis=3)

    def test_stack_out_of_bounds2(self):
        a = testing.shaped_arange((2, 3), cupy)
        with self.assertRaises(numpy.AxisError):
            return cupy.stack([a, a], axis=3)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_stack_out(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((3, 4), xp, dtype)
        c = testing.shaped_arange((3, 4), xp, dtype)
        out = xp.zeros((3, 3, 4), dtype=dtype)
        xp.stack((a, b, c), axis=1, out=out)
        return out

    @testing.numpy_cupy_array_equal()
    def test_stack_out_same_kind(self, xp):
        a = testing.shaped_arange((3, 4), xp, xp.float64)
        b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
        c = testing.shaped_arange((3, 4), xp, xp.float64)
        out = xp.zeros((3, 3, 4), dtype=xp.float32)
        xp.stack((a, b, c), axis=1, out=out)
        return out

    def test_stack_out_invalid_shape(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((3, 3, 10), dtype=xp.float64)
            with pytest.raises(ValueError):
                xp.stack((a, b, c), axis=1, out=out)

    def test_stack_out_invalid_shape_2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((3, 3, 3, 10), dtype=xp.float64)
            with pytest.raises(ValueError):
                xp.stack((a, b, c), axis=1, out=out)

    def test_stack_out_invalid_dtype(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 4), xp, xp.float64)
            b = testing.shaped_reverse_arange((3, 4), xp, xp.float64)
            c = testing.shaped_arange((3, 4), xp, xp.float64)
            out = xp.zeros((3, 3, 4), dtype=xp.int64)
            with pytest.raises(TypeError):
                xp.stack((a, b, c), axis=1, out=out)
