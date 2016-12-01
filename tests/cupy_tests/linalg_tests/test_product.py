import unittest

from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [
        ((2, 3, 4), (3, 4, 2)),
        ((1, 1), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (2, 1)),
        ((2, 1), (1, 1)),
        ((1, 2), (2, 3)),
        ((2, 1), (1, 3)),
        ((2, 3), (3, 1)),
        ((2, 3), (3, 4)),
        ((0, 3), (3, 4)),
        ((2, 3), (3, 0)),
        ((0, 3), (3, 0)),
        ((3, 0), (0, 4)),
        ((2, 3, 0), (3, 0, 2)),
        ((0, 0), (0, 0)),
        ((3,), (3,)),
        ((2,), (2, 4)),
        ((4, 2), (2,)),
    ],
    'trans_a': [True, False],
    'trans_b': [True, False],
}))
@testing.gpu
class TestDot(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot(self, xp, dtype_a, dtype_b):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.dot(a, b)

    @testing.for_float_dtypes(name='dtype_a')
    @testing.for_float_dtypes(name='dtype_b')
    @testing.for_float_dtypes(name='dtype_c')
    @testing.numpy_cupy_allclose(accept_error=ValueError)
    def test_dot_with_out(self, xp, dtype_a, dtype_b, dtype_c):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        if a.ndim == 0 or b.ndim == 0:
            shape_c = shape_a + shape_b
        else:
            shape_c = shape_a[:-1] + shape_b[:-2] + shape_b[-1:]
        c = xp.empty(shape_c, dtype=dtype_c)
        out = xp.dot(a, b, out=c)
        self.assertIs(out, c)
        return c


@testing.parameterize(*testing.product({
    'shape': [
        ((), ()),
        ((), (2, 4)),
        ((4, 2), ()),
    ],
    'trans_a': [True, False],
    'trans_b': [True, False],
}))
@testing.gpu
class TestDotFor0Dim(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot(self, xp, dtype_a, dtype_b):
        shape_a, shape_b = self.shape
        if self.trans_a:
            a = testing.shaped_arange(shape_a[::-1], xp, dtype_a).T
        else:
            a = testing.shaped_arange(shape_a, xp, dtype_a)
        if self.trans_b:
            b = testing.shaped_arange(shape_b[::-1], xp, dtype_b).T
        else:
            b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.dot(a, b)


@testing.gpu
class TestProduct(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec1(self, xp, dtype):
        a = testing.shaped_arange((2,), xp, dtype)
        b = testing.shaped_arange((2,), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec2(self, xp, dtype):
        a = testing.shaped_arange((2,), xp, dtype)
        b = testing.shaped_arange((2, 1), xp, dtype)
        return xp.dot(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_vec3(self, xp, dtype):
        a = testing.shaped_arange((1, 2), xp, dtype)
        b = testing.shaped_arange((2,), xp, dtype)
        return xp.dot(a, b)

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
    @testing.numpy_cupy_raises()
    def test_transposed_dot_with_out2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(1, 0, 2)
        b = testing.shaped_arange((4, 2, 3), xp, dtype).transpose(2, 0, 1)
        c = xp.ndarray((3, 2, 3, 2)[::-1], dtype=dtype).T
        # Only C-contiguous array is acceptable
        xp.dot(a, b, out=c)

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
