import unittest

import numpy

from cupy import testing


class TestEinSumError(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_no_arguments(self, xp):
        xp.einsum()

    @testing.numpy_cupy_raises()
    def test_one_argument(self, xp):
        xp.einsum('')

    @testing.numpy_cupy_raises()
    def test_not_string_subject(self, xp):
        xp.einsum(0, 0)

    @testing.numpy_cupy_raises()
    def test_bad_argument(self, xp):
        xp.einsum('', 0, bad_arg=0)

    @testing.numpy_cupy_raises()
    def test_too_many_operands(self, xp):
        xp.einsum('', 0, 0)

    @testing.numpy_cupy_raises()
    def test_too_few_operands(self, xp):
        xp.einsum(',', 0)

    @testing.numpy_cupy_raises()
    def test_many_dimension1(self, xp):
        xp.einsum('i', 0)

    @testing.numpy_cupy_raises()
    def test_many_dimension2(self, xp):
        xp.einsum('ij', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_too_few_dimension(self, xp):
        xp.einsum('i->i', xp.arange(6).reshape(2, 3))

    @testing.numpy_cupy_raises()
    def test_invalid_char1(self, xp):
        xp.einsum('i%', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_invalid_char2(self, xp):
        xp.einsum('j$', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_invalid_char3(self, xp):
        xp.einsum('i->&', xp.array([0, 0]))

    # output subscripts must appear in inumpy.t
    @testing.numpy_cupy_raises()
    def test_invalid_output_subscripts1(self, xp):
        xp.einsum('i->ij', xp.array([0, 0]))

    # output subscripts may only be specified once
    @testing.numpy_cupy_raises()
    def test_invalid_output_subscripts2(self, xp):
        xp.einsum('ij->jij', xp.array([[0, 0], [0, 0]]))

    # output subscripts must not incrudes comma
    @testing.numpy_cupy_raises()
    def test_invalid_output_subscripts3(self, xp):
        xp.einsum('ij->i,j', xp.array([[0, 0], [0, 0]]))

    # dimensions much match when being collapsed
    @testing.numpy_cupy_raises()
    def test_invalid_diagonal1(self, xp):
        xp.einsum('ii', xp.arange(6).reshape(2, 3))

    @testing.numpy_cupy_raises()
    def test_invalid_diagonal2(self, xp):
        xp.einsum('ii->', xp.arange(6).reshape(2, 3))

    # invalid -> operator
    @testing.numpy_cupy_raises()
    def test_invalid_arrow1(self, xp):
        xp.einsum('i-i', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_invalid_arrow2(self, xp):
        xp.einsum('i>i', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_invalid_arrow3(self, xp):
        xp.einsum('i->->i', xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_invalid_arrow4(self, xp):
        xp.einsum('i-', xp.array([0, 0]))


class TestEinSum(unittest.TestCase):
    # Avoid overflow
    skip_dtypes = (numpy.bool_, numpy.int8, numpy.uint8)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_do_nothing(self, xp, dtype):
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transpose(self, xp, dtype):
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ji', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diagonal(self, xp, dtype):
        shape_a = (3, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ii->i', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diagonal_3d_1(self, xp, dtype):
        shape_a = (3, 3, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('jii->ij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diagonal_3d_2(self, xp, dtype):
        shape_a = (3, 3, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('iji->ij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diagonal_3d_3(self, xp, dtype):
        shape_a = (3, 3, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('iii->i', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_swap_axes_1(self, xp, dtype):
        shape_a = (2, 3, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijk->jik', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_swap_axes_2(self, xp, dtype):
        shape_a = (2, 3, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijk->kij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum(self, xp, dtype):
        shape_a = (3,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('i->', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trace(self, xp, dtype):
        shape_a = (3, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ii', a)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_outer(self, xp, dtype_a, dtype_b):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype_a) + 1
        shape_b = (3,)
        b = testing.shaped_arange(shape_b, xp, dtype_b) + 1
        return xp.einsum('i,j', a, b)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot_matvec_1(self, xp, dtype_a, dtype_b):
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        shape_b = (3,)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.einsum('ij,j', a, b)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot_matvec_2(self, xp, dtype_a, dtype_b):
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        shape_b = (2,)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.einsum('ji,j', a, b)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_dot_matmat(self, xp, dtype_a, dtype_b):
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        shape_b = (3, 4)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.einsum('ij,jk', a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_dot_matmatmat(self, xp, dtype):
        if dtype in self.skip_dtypes:
            return xp.array([])
        shape_a = (2, 3)
        a = testing.shaped_arange(shape_a, xp, dtype)
        shape_b = (3, 4)
        b = testing.shaped_arange(shape_b, xp, dtype)
        shape_c = (4, 5)
        c = testing.shaped_arange(shape_c, xp, dtype)
        return xp.einsum('ij,jk,kl', a, b, c).astype(numpy.float32)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_tensordot(self, xp, dtype_a, dtype_b):
        if dtype_a in self.skip_dtypes or dtype_b in self.skip_dtypes:
            return xp.array([])
        shape_a = (3, 4, 2)
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        shape_b = (4, 3, 2)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.einsum('ijk, jil -> kl', a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_1(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum(',i->', 3, a))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_2(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum('i,->', a, 4))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transpose_and_diagonal(self, xp, dtype):
        shape_a = (2, 2, 2, 2)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijkj->kij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trase_and_tensordot_and_diagobal(self, xp, dtype):
        if dtype in self.skip_dtypes:
            return xp.array([])
        shape_a = (2, 3, 2, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        shape_b = (3, 2, 2)
        b = testing.shaped_arange(shape_b, xp, dtype)
        return xp.einsum('ijil,jkk->kj', a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trase_4d_1(self, xp, dtype):
        shape_a = (2, 2, 2, 2)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijij->ij', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trase_4d_2(self, xp, dtype):
        shape_a = (2, 2, 2, 2)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('jiji->ji', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transpose_3d_1(self, xp, dtype):
        shape_a = (2, 3, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijk->ikj', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transpose_3d_2(self, xp, dtype):
        shape_a = (2, 3, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('ijk->jik', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_transpose_3d_3(self, xp, dtype):
        shape_a = (2, 3, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.einsum('kji->ikj', a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_triple_product_1(self, xp, dtype):
        shape_a = (2, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        shape_b = (2, 3)
        b = testing.shaped_arange(shape_b, xp, dtype)
        shape_c = (2,)
        c = testing.shaped_arange(shape_c, xp, dtype)
        return xp.einsum('ij,ik,i->ijk', a, b, c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_triple_product_2(self, xp, dtype):
        shape_a = (2, 4)
        a = testing.shaped_arange(shape_a, xp, dtype)
        shape_b = (3, 2)
        b = testing.shaped_arange(shape_b, xp, dtype)
        shape_c = (2,)
        c = testing.shaped_arange(shape_c, xp, dtype)
        return xp.einsum('ij,ki,i->jk', a, b, c)
