import unittest

import numpy

import cupy_testing as testing


"""
class TestA(unittest.TestCase):

    shape_a = (1, 1)
    shape_b = (3, 4)
    shape_c = (1, 1)
    subscripts = 'ij,jk,kl'
    optimize = True

    @testing.for_all_dtypes_combination(['dtype_x', 'dtype_y'], no_float16=True)
    # @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_ternary(self, dtype_x, dtype_y):
        a = testing.shaped_arange(self.shape_a, numpy, dtype_x)
        b = testing.shaped_arange(self.shape_b, numpy, dtype_x)
        c = testing.shaped_arange(self.shape_c, numpy, dtype_y)
        return numpy.einsum(self.subscripts, a, b, c, optimize=self.optimize)


a = np.array([[1.+1.j]])
b = np.array([[ 1. +1.j,  2. +2.j,  3. +3.j,  4. +4.j],
       [ 5. +5.j,  6. +6.j,  7. +7.j,  8. +8.j],
       [ 9. +9.j, 10.+10.j, 11.+11.j, 12.+12.j]])
c = np.array([[1]], dtype=np.int32)
np.einsum('ij,jk,kl', a, b, c, optimize=True)
# This raises error, due to numpy issue #10930.
"""


class TestB(unittest.TestCase):

    shape_a = (3, 4, 2)
    shape_b = (1, 1, 2)
    subscripts = 'ijk,jil->kl'
    # optimize = True

    @testing.for_all_dtypes_combination(['dtype_a', 'dtype_b'])
    # @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_np(self, dtype_a, dtype_b):
        a = testing.shaped_arange(self.shape_a, numpy, dtype_a)
        b = testing.shaped_arange(self.shape_b, numpy, dtype_b)
        # c = testing.shaped_arange(self.shape_c, numpy, dtype_y)
        return numpy.einsum(self.subscripts, a, b)


class TestC(unittest.TestCase):
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test1(self, xp):
        shape_a = (1, 1, 1, 1, 1, 1)
        dtype_a = numpy.complex128
        shape_b = (2, 3, 2, 2)
        dtype_b = numpy.int8
        subscripts = '...lmn,lmno->...o'
        a = testing.shaped_arange(shape_a, xp, dtype_a)
        b = testing.shaped_arange(shape_b, xp, dtype_b)
        return xp.einsum(subscripts, a, b, optimize=False)


def _dec_shape(shape, dec):
    return tuple(1 if s == 1 else max(0, s - dec) for s in shape)


def _rand1_shape(shape, prob):
    # return tuple(1 if numpy.random.rand() < prob else s for s in shape)
    table = {}
    new_shape = []
    for s in shape:
        if s not in table:
            table[s] = 1 if numpy.random.rand() < prob else s
        new_shape.append(table[s])
    return tuple(new_shape)


def augument_einsum_testcases(*params):
    # rs = numpy.random.RandomState()
    for dec in range(3):
        for drop in [0, 0.2, 0.8]:
            for param in params:
                param_new = param.copy()
                for k in param.keys():
                    if k.startswith('shape_'):
                        param_new[k] = \
                            _rand1_shape(_dec_shape(param[k], dec), drop)
                param_new['_raw_params'] = {
                    'orig': param,
                    'dec': dec,
                    'drop': drop,
                }
                yield param_new


class TestEinSumError(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_irregular_ellipsis1(self, xp):
        xp.einsum('..', xp.zeros((2, 2, 2)))

    @testing.numpy_cupy_raises()
    def test_irregular_ellipsis2(self, xp):
        xp.einsum('...i...', xp.zeros((2, 2, 2)))

    @testing.numpy_cupy_raises()
    def test_irregular_ellipsis3(self, xp):
        xp.einsum('i...->...i...', xp.zeros((2, 2, 2)))

    @testing.numpy_cupy_raises()
    def test_irregular_ellipsis4(self, xp):
        xp.einsum('...->', xp.zeros((2, 2, 2)))

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
    def test_too_many_dimension3(self, xp):
        xp.einsum('ijk...->...', xp.arange(6).reshape(2, 3))

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

    @testing.numpy_cupy_raises()
    def test_invalid_diagonal3(self, xp):
        xp.einsum('ii', xp.arange(3).reshape(1, 3))

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


@testing.parameterize(*augument_einsum_testcases(
    {'shape_a': (2, 3), 'subscripts': 'ij'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': '...'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'ji'},  # transpose
    {'shape_a': (3, 3), 'subscripts': 'ii->i'},  # diagonal 2d
    {'shape_a': (3, 3, 3), 'subscripts': 'jii->ij'},  # partial diagonal 3d
    {'shape_a': (3, 3, 3), 'subscripts': 'iji->ij'},  # partial diagonal 3d
    {'shape_a': (3, 3, 3), 'subscripts': '...ii->...i'},  # partial diagonal 3d
    {'shape_a': (3, 3, 3), 'subscripts': 'iii->i'},  # diagonal 3d
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->jik'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->kij'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'ijk->ikj'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'kji->ikj'},  # swap axes
    {'shape_a': (2, 3, 4), 'subscripts': 'j...i->i...j'},  # swap axes
    {'shape_a': (3,), 'subscripts': 'i->'},  # sum
    {'shape_a': (3, 3), 'subscripts': 'ii'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'ijkj->kij'},
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'ijij->ij'},
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'jiji->ij'},
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'ii...->...'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'i...i->...'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': '...ii->...'},  # trace
    {'shape_a': (2, 2, 2, 2), 'subscripts': 'j...i->...'},  # sum

    {'shape_a': (2, 3), 'subscripts': 'ij->ij...'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'ij->i...j'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'ij->...ij'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'ij...->ij'},  # do nothing
    {'shape_a': (2, 3), 'subscripts': 'i...j->ij'},  # do nothing

    {'shape_a': (), 'subscripts': ''},  # do nothing
    {'shape_a': (), 'subscripts': '->'},  # do nothing
))
class TestEinSumUnaryOperation(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_unary(self, xp, dtype):
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        return xp.einsum(self.subscripts, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_einsum_unary_views(self, xp, dtype):
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        b = xp.einsum(self.subscripts, a)

        return b.base is not None, b.base is a

    @testing.for_all_dtypes_combination(
        ['dtype_a', 'dtype_out'],
        no_complex=True)  # avoid ComplexWarning
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_unary_dtype(self, xp, dtype_a, dtype_out):
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        return xp.einsum(self.subscripts, a, dtype=dtype_out, casting='unsafe')


class TestEinSumUnaryOperationWithScalar(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_int(self, xp, dtype):
        return xp.asarray(xp.einsum('', 2, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_float(self, xp, dtype):
        return xp.asarray(xp.einsum('', 2.0, dtype=dtype))


@testing.parameterize(*augument_einsum_testcases(
    # outer
    {'shape_a': (2,), 'shape_b': (3,),
     'subscripts': 'i,j'},
    # dot matvec
    {'shape_a': (2, 3), 'shape_b': (3,),
     'subscripts': 'ij,j'},
    {'shape_a': (2, 3), 'shape_b': (2,),
     'subscripts': 'ij,i'},
    # dot matmat
    {'shape_a': (2, 3), 'shape_b': (3, 4),
     'subscripts': 'ij,jk'},
    # tensordot
    {'shape_a': (3, 4, 2), 'shape_b': (4, 3, 2),
     'subscripts': 'ijk, jil -> kl'},
    {'shape_a': (3, 4, 2), 'shape_b': (4, 2, 3),
     'subscripts': 'i...,...k->ki...'},
    {'shape_a': (3, 4, 2), 'shape_b': (4, 3, 2),
     'subscripts': 'ij...,ji...->i...'},
    # trace and tensordot and diagonal
    {'shape_a': (2, 3, 2, 4), 'shape_b': (3, 2, 2),
     'subscripts': 'ijil,jkk->kj'},
    {'shape_a': (2, 4, 2, 3), 'shape_b': (3, 2, 4),
     'subscripts': 'i...ij,ji...->...j'},
    # broadcast
    {'shape_a': (2, 3, 4), 'shape_b': (3,),
     'subscripts': 'ij...,j...->ij...'},
    {'shape_a': (2, 3, 4), 'shape_b': (3,),
     'subscripts': 'ij...,...j->ij...'},
    {'shape_a': (2, 3, 4), 'shape_b': (3,),
     'subscripts': 'ij...,j->ij...'},
    {'shape_a': (4, 3), 'shape_b': (3, 2),
     'subscripts': 'ik...,k...->i...'},
    {'shape_a': (4, 3), 'shape_b': (3, 2),
     'subscripts': 'ik...,...kj->i...j'},
    {'shape_a': (4, 3), 'shape_b': (3, 2),
     'subscripts': '...k,kj'},
    {'shape_a': (4, 3), 'shape_b': (3, 2),
     'subscripts': 'ik,k...->i...'},
    {'shape_a': (2, 3, 4, 5), 'shape_b': (4,),
     'subscripts': 'ijkl,k'},
    {'shape_a': (2, 3, 4, 5), 'shape_b': (4,),
     'subscripts': '...kl,k'},
    {'shape_a': (2, 3, 4, 5), 'shape_b': (4,),
     'subscripts': '...kl,k...'},
    {'shape_a': (1, 1, 1, 2, 3, 2), 'shape_b': (2, 3, 2, 2),
     'subscripts': '...lmn,lmno->...o'},
))
class TestEinSumBinaryOperation(unittest.TestCase):
    @testing.for_all_dtypes_combination(
        ['dtype_a', 'dtype_b'],
        no_float16=True)  # Avoid numpy issue #10899
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_binary(self, xp, dtype_a, dtype_b):
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        b = testing.shaped_arange(self.shape_b, xp, dtype_b)
        return xp.einsum(self.subscripts, a, b)


class TestEinSumBinaryOperationWithScalar(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_scalar_1(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum(',i->', 3, a))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_scalar_2(self, xp, dtype):
        shape_a = (2,)
        a = testing.shaped_arange(shape_a, xp, dtype)
        return xp.asarray(xp.einsum('i,->', a, 4))


@testing.parameterize(*augument_einsum_testcases(
    {'shape_a': (2, 3), 'shape_b': (3, 4), 'shape_c': (4, 5),
     'subscripts': 'ij,jk,kl'},
    {'shape_a': (2, 4), 'shape_b': (2, 3), 'shape_c': (2,),
     'subscripts': 'ij,ik,i->ijk'},
    {'shape_a': (2, 4), 'shape_b': (3, 2), 'shape_c': (2,),
     'subscripts': 'ij,ki,i->jk'},
    {'shape_a': (2, 3, 4), 'shape_b': (2,), 'shape_c': (3, 4, 2),
     'subscripts': 'i...,i,...i->...i'},
))
class TestEinSumTernaryOperation(unittest.TestCase):
    @testing.for_all_dtypes_combination(
        ['dtype_a', 'dtype_b', 'dtype_c'],
        no_float16=True)  # Avoid numpy issue #10899
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_ternary(self, xp, dtype_a, dtype_b, dtype_c):
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        b = testing.shaped_arange(self.shape_b, xp, dtype_b)
        c = testing.shaped_arange(self.shape_c, xp, dtype_c)

        out = xp.einsum(self.subscripts, a, b, c, optimize=False)

        if xp is not numpy:  # Avoid numpy issues #11059, #11060
            for optimize in [
                    True,  # 'greedy'
                    'optimal',
                    ['einsum_path', (0, 1), (0, 1)],
                    ['einsum_path', (0, 2), (0, 1)],
                    ['einsum_path', (1, 2), (0, 1)],
            ]:
                optimized_out = xp.einsum(
                    self.subscripts, a, b, c, optimize=optimize)
                testing.assert_allclose(optimized_out, out)
        return out


testing.run_module(__name__, __file__)
