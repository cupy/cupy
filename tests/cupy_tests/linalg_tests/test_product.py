import unittest

from cupy import testing


@testing.gpu
class TestProduct(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot2(self, xp, dtype):
        a = testing.shaped_arange((4, 1), xp, dtype)
        b = testing.shaped_arange((1, 3), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot3(self, xp, dtype):
        a = testing.shaped_arange((2, 1), xp, dtype).T
        b = testing.shaped_arange((2, 1), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot4(self, xp, dtype):
        a = testing.shaped_arange((1, 2), xp, dtype)
        b = testing.shaped_arange((1, 2), xp, dtype).T
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot5(self, xp, dtype):
        a = testing.shaped_arange((2, 1), xp, dtype).T
        b = testing.shaped_arange((1, 2), xp, dtype).T
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot6(self, xp, dtype):
        a = testing.shaped_arange((10, 2), xp, dtype)
        b = testing.shaped_arange((2, 10), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 4, 2), xp, dtype)
        c = xp.ndarray((2, 3, 3, 2), dtype=dtype)
        xp.dot(a, b, out=c)
        return c

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(0, 2, 1)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_dot_with_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(2, 0, 1)
        c = xp.ndarray((3, 2, 3, 2), dtype=dtype)
        xp.dot(a, b, out=c)
        return c

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array1(self, xp, dtype):
        a = testing.shaped_arange((3, 1), xp, dtype)
        b = xp.array([[2]], dtype=dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_with_single_elem_array2(self, xp, dtype):
        a = xp.array([[2]], dtype=dtype)
        b = testing.shaped_arange((1, 3), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_vdot(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_reverse_arange((5,), xp, dtype)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_vdot(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xp, dtype)[::-1]
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_vdot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((2, 2, 2, 3), xp, dtype)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_multidim_vdot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange(
            (2, 2, 2, 3), xp, dtype).transpose(1, 3, 0, 2)
        return xp.vdot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_inner(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_reverse_arange((5,), xp, dtype)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_inner(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)[::-1]
        b = testing.shaped_reverse_arange((5,), xp, dtype)[::-1]
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_inner(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 2, 4), xp, dtype)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_higher_order_inner(self, xp, dtype):
        a = testing.shaped_arange((2, 4, 3), xp, dtype).transpose(2, 0, 1)
        b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(1, 2, 0)
        return xp.inner(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_outer(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_arange((4,), xp, dtype)
        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_reversed_outer(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        b = testing.shaped_arange((4,), xp, dtype)
        return xp.outer(a[::-1], b[::-1])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_multidim_outer(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_arange((4, 5), xp, dtype)
        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 3, 2), xp, dtype).transpose(2, 0, 1)
        return xp.tensordot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_int_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        b = testing.shaped_arange((3, 4, 5, 2), xp, dtype)
        return xp.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_int_axes(self, xp, dtype):
        a = testing.shaped_arange(
            (2, 3, 4, 5), xp, dtype).transpose(2, 0, 3, 1)
        b = testing.shaped_arange(
            (5, 4, 3, 2), xp, dtype).transpose(3, 0, 2, 1)
        return xp.tensordot(a, b, axes=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_with_list_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        b = testing.shaped_arange((3, 5, 4, 2), xp, dtype)
        return xp.tensordot(a, b, axes=([3, 2, 1], [1, 2, 0]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transposed_tensordot_with_list_axes(self, xp, dtype):
        a = testing.shaped_arange(
            (2, 3, 4, 5), xp, dtype).transpose(2, 0, 3, 1)
        b = testing.shaped_arange(
            (3, 5, 4, 2), xp, dtype).transpose(3, 0, 2, 1)
        return xp.tensordot(a, b, axes=([2, 0, 3], [3, 2, 1]))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_tensordot_zero_dim(self, xp, dtype):
        a = xp.array(2, dtype=dtype)
        b = testing.shaped_arange((3, 4, 2), xp, dtype)
        return xp.tensordot(a, b, axes=0)
