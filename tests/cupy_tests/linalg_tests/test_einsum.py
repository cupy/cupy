import unittest

import numpy

import cupy
from cupy import testing


class TestEinSumError(unittest.TestCase):

    # '...' ellipsis is not supported.
    def test_not_supported_ellipsis(self):
        with self.assertRaises(TypeError):
            cupy.einsum('...', 0)

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
    def test_too_many_operands1(self, xp):
        xp.einsum('', 0, 0)

    @testing.numpy_cupy_raises()
    def test_too_many_operands2(self, xp):
        xp.einsum('i,j', xp.array([0, 0]), xp.array([0, 0]), xp.array([0, 0]))

    @testing.numpy_cupy_raises()
    def test_too_few_operands1(self, xp):
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


@testing.parameterize(
    {'shape_a': (2, 3), 'subscripts': 'ij'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'ij'},  # transpose
    {'shape_a': (3, 3), 'subscripts': 'ii->i'},  # diagonal 2d
    {'shape_a': (3, 3, 3), 'subscripts': 'jii->ij'},  # partial diagonal 3d
    {'shape_a': (3, 3, 3), 'subscripts': 'iji->ij'},  # partial diagonal 3d
    {'shape_a': (3, 3, 3), 'subscripts': 'iii->i'},  # diagonal 3d
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->jik'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->kij'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->ikj'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'kji->ikj'},  # swap axes
    {'shape_a': (3,), 'subscripts': 'i->'},  # sum
    {'shape_a': (3, 3), 'subscripts': 'ii'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'ijkj->kij'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'ijij->ij'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'jiji->ij'},  # trace
)
class TestEinSumUnaryOperation(unittest.TestCase):
    # Avoid overflow
    skip_dtypes = (numpy.bool_, numpy.int8, numpy.uint8)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_einsum_unary(self, xp, dtype):
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        return xp.einsum(self.subscripts, a)


@testing.parameterize(
    # outer
    {'shape_a': (2,), 'shape_b': (3,),
     'subscripts': 'i,j', 'skip_overflow': False},
    # dot matvec
    {'shape_a': (2, 3), 'shape_b': (3,),
     'subscripts': 'ij,j', 'skip_overflow': False},
    {'shape_a': (2, 3), 'shape_b': (2,),
     'subscripts': 'ij,i', 'skip_overflow': False},
    # dot matmat
    {'shape_a': (2, 3), 'shape_b': (3, 4),
     'subscripts': 'ij,jk', 'skip_overflow': False},
    # tensordot
    {'shape_a': (3, 4, 2), 'shape_b': (4, 3, 2),
     'subscripts': 'ijk, jil -> kl', 'skip_overflow': True},
    # trace and tensordot and diagonal
    {'shape_a': (2, 3, 2, 4), 'shape_b': (3, 2, 2),
     'subscripts': 'ijil,jkk->kj', 'skip_overflow': True},
)
class TestEinSumBinaryOperation(unittest.TestCase):
    skip_dtypes = (numpy.bool_, numpy.int8, numpy.uint8)

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    @testing.numpy_cupy_allclose()
    def test_einsum_binary(self, xp, dtype_a, dtype_b):
        if self.skip_overflow and (dtype_a in self.skip_dtypes or
                                   dtype_b in self.skip_dtypes):
            return xp.array([])
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        b = testing.shaped_arange(self.shape_b, xp, dtype_b)
        return xp.einsum(self.subscripts, a, b)


class TestEinSumBinaryOperationWithScalar(unittest.TestCase):
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


@testing.parameterize(
    {'shape_a': (2, 3), 'shape_b': (3, 4), 'shape_c': (4, 5),
     'subscripts': 'ij,jk,kl', 'skip_overflow': True},
    {'shape_a': (2, 4), 'shape_b': (2, 3), 'shape_c': (2,),
     'subscripts': 'ij,ik,i->ijk', 'skip_overflow': False},
    {'shape_a': (2, 4), 'shape_b': (3, 2), 'shape_c': (2,),
     'subscripts': 'ij,ki,i->jk', 'skip_overflow': False},
)
class TestEinSumTernaryOperation(unittest.TestCase):
    skip_dtypes = (numpy.bool_, numpy.int8, numpy.uint8)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_ternary(self, xp, dtype):
        if self.skip_overflow and dtype in self.skip_dtypes:
            return xp.array([])
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        b = testing.shaped_arange(self.shape_b, xp, dtype)
        c = testing.shaped_arange(self.shape_c, xp, dtype)
        return xp.einsum(self.subscripts, a, b, c).astype(numpy.float32)
