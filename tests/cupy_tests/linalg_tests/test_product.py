import unittest

from cupy import testing


@testing.gpu
class TestProduct(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_arange((3, 4, 2), xpy, dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot2(self, xpy, dtype):
        a = testing.shaped_arange((4, 1), xpy, dtype)
        b = testing.shaped_arange((1, 3), xpy, dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot3(self, xpy, dtype):
        a = testing.shaped_arange((2, 1), xpy, dtype).T
        b = testing.shaped_arange((2, 1), xpy, dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot4(self, xpy, dtype):
        a = testing.shaped_arange((1, 2), xpy, dtype)
        b = testing.shaped_arange((1, 2), xpy, dtype).T
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot5(self, xpy, dtype):
        a = testing.shaped_arange((2, 1), xpy, dtype).T
        b = testing.shaped_arange((1, 2), xpy, dtype).T
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot6(self, xpy, dtype):
        a = testing.shaped_arange((10, 2), xpy, dtype)
        b = testing.shaped_arange((2, 10), xpy, dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_out(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_arange((3, 4, 2), xpy, dtype)
        c = xpy.ndarray((2, 3, 3, 2), dtype=dtype)
        xpy.dot(a, b, out=c)
        return c

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((2, 3, 4), xpy, dtype).transpose(0, 2, 1)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot_with_out(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 2, 3), xpy, dtype).transpose(2, 0, 1)
        c = xpy.ndarray((3, 2, 3, 2), dtype=dtype)
        xpy.dot(a, b, out=c)
        return c

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array1(self, xpy, dtype):
        a = testing.shaped_arange((3, 1), xpy, dtype)
        b = xpy.array([[2]], dtype=dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array2(self, xpy, dtype):
        a = xpy.array([[2]], dtype=dtype)
        b = testing.shaped_arange((1, 3), xpy, dtype)
        return xpy.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vdot(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)
        b = testing.shaped_reverse_arange((5,), xpy, dtype)
        return xpy.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_vdot(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xpy, dtype)[::-1]
        return xpy.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_vdot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_arange((2, 2, 2, 3), xpy, dtype)
        return xpy.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_multidim_vdot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange(
            (2, 2, 2, 3), xpy, dtype).transpose(1, 3, 0, 2)
        return xpy.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_inner(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)
        b = testing.shaped_reverse_arange((5,), xpy, dtype)
        return xpy.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_inner(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xpy, dtype)[::-1]
        return xpy.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_inner(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_arange((3, 2, 4), xpy, dtype)
        return xpy.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_higher_order_inner(self, xpy, dtype):
        a = testing.shaped_arange((2, 4, 3), xpy, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange((4, 2, 3), xpy, dtype).transpose(1, 2, 0)
        return xpy.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_outer(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)
        b = testing.shaped_arange((4,), xpy, dtype)
        return xpy.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_outer(self, xpy, dtype):
        a = testing.shaped_arange((5,), xpy, dtype)
        b = testing.shaped_arange((4,), xpy, dtype)
        return xpy.outer(a[::-1], b[::-1])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_outer(self, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        b = testing.shaped_arange((4, 5), xpy, dtype)
        return xpy.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_arange((3, 4, 5), xpy, dtype)
        return xpy.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 3, 2), xpy, dtype).transpose(2, 0, 1)
        return xpy.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_int_axes(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xpy, dtype)
        b = testing.shaped_arange((3, 4, 5, 2), xpy, dtype)
        return xpy.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_int_axes(self, xpy, dtype):
        a = testing.shaped_arange(
            (2, 3, 4, 5), xpy, dtype).transpose(2, 0, 3, 1)
        b = testing.shaped_arange(
            (5, 4, 3, 2), xpy, dtype).transpose(3, 0, 2, 1)
        return xpy.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_list_axes(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xpy, dtype)
        b = testing.shaped_arange((3, 5, 4, 2), xpy, dtype)
        return xpy.tensordot(a, b, axes=([3, 2, 1], [1, 2, 0]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_list_axes(self, xpy, dtype):
        a = testing.shaped_arange(
            (2, 3, 4, 5), xpy, dtype).transpose(2, 0, 3, 1)
        b = testing.shaped_arange(
            (3, 5, 4, 2), xpy, dtype).transpose(3, 0, 2, 1)
        return xpy.tensordot(a, b, axes=([2, 0, 3], [3, 2, 1]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_zero_dim(self, xpy, dtype):
        a = xpy.array(2, dtype=dtype)
        b = testing.shaped_arange((3, 4, 2), xpy, dtype)
        return xpy.tensordot(a, b, axes=0)
