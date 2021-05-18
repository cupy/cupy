import unittest
import warnings

import numpy
import pytest

import cupy
from cupy import testing


def _dec_shape(shape, dec):
    # Test smaller shape
    return tuple(1 if s == 1 else max(0, s - dec) for s in shape)


def _rand1_shape(shape, prob):
    # Test broadcast
    # If diagonals are "broadcasted" we can simply:
    # return tuple(1 if numpy.random.rand() < prob else s for s in shape)
    table = {}
    new_shape = []
    for s in shape:
        if s not in table:
            table[s] = 1 if numpy.random.rand() < prob else s
        new_shape.append(table[s])
    return tuple(new_shape)


def augument_einsum_testcases(*params):
    """Modify shapes in einsum tests

    Shape parameter should be starts with 'shape_'.
    The original parameter is stored as '_raw_params'.

    Args:
        params (sequence of dicts)

    Yields:
        dict: parameter with modified shapes.

    """
    for dec in range(3):
        for drop in [False, True]:
            for param in params:
                param_new = param.copy()
                for k in param.keys():
                    if k.startswith('shape_'):
                        new_shape = _dec_shape(param[k], dec)
                        if drop:
                            prob = numpy.random.rand()
                            new_shape = _rand1_shape(new_shape, prob)
                        param_new[k] = new_shape
                param_new['_raw_params'] = {
                    'orig': param,
                    'dec': dec,
                    'drop': drop,
                }
                yield param_new


class TestEinSumError(unittest.TestCase):

    def test_irregular_ellipsis1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('..', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...i...', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i...->...i...', xp.zeros((2, 2, 2)))

    def test_irregular_ellipsis4(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...->', xp.zeros((2, 2, 2)))

    def test_no_arguments(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum()

    def test_one_argument(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('')

    def test_not_string_subject(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.einsum(0, 0)

    def test_bad_argument(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.einsum('', 0, bad_arg=0)

    def test_too_many_operands1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('', 0, 0)

    def test_too_many_operands2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(
                    'i,j',
                    xp.array([0, 0]),
                    xp.array([0, 0]),
                    xp.array([0, 0]))

    def test_too_few_operands1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(',', 0)

    def test_too_many_dimension1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i', 0)

    def test_too_many_dimension2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij', xp.array([0, 0]))

    def test_too_many_dimension3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ijk...->...', xp.arange(6).reshape(2, 3))

    def test_too_few_dimension(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->i', xp.arange(6).reshape(2, 3))

    def test_invalid_char1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i%', xp.array([0, 0]))

    def test_invalid_char2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('j$', xp.array([0, 0]))

    def test_invalid_char3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->&', xp.array([0, 0]))

    # output subscripts must appear in inumpy.t
    def test_invalid_output_subscripts1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->ij', xp.array([0, 0]))

    # output subscripts may only be specified once
    def test_invalid_output_subscripts2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij->jij', xp.array([[0, 0], [0, 0]]))

    # output subscripts must not incrudes comma
    def test_invalid_output_subscripts3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ij->i,j', xp.array([[0, 0], [0, 0]]))

    # dimensions much match when being collapsed
    def test_invalid_diagonal1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii', xp.arange(6).reshape(2, 3))

    def test_invalid_diagonal2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii->', xp.arange(6).reshape(2, 3))

    def test_invalid_diagonal3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('ii', xp.arange(3).reshape(1, 3))

    def test_dim_mismatch_char1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i,i', xp.arange(2), xp.arange(3))

    def test_dim_mismatch_ellipsis1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('...,...', xp.arange(2), xp.arange(3))

    def test_dim_mismatch_ellipsis2(self):
        for xp in (numpy, cupy):
            a = xp.arange(12).reshape(2, 3, 2)
            with pytest.raises(ValueError):
                xp.einsum('i...,...i', a, a)

    def test_dim_mismatch_ellipsis3(self):
        for xp in (numpy, cupy):
            a = xp.arange(12).reshape(2, 3, 2)
            with pytest.raises(ValueError):
                xp.einsum('...,...', a, a[:, :2])

    # invalid -> operator
    def test_invalid_arrow1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i-i', xp.array([0, 0]))

    def test_invalid_arrow2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i>i', xp.array([0, 0]))

    def test_invalid_arrow3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i->->i', xp.array([0, 0]))

    def test_invalid_arrow4(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum('i-', xp.array([0, 0]))


class TestListArgEinSumError(unittest.TestCase):

    @testing.with_requires('numpy>=1.19')
    def test_invalid_sub1(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.einsum(xp.arange(2), [None])

    def test_invalid_sub2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], [1])

    def test_invalid_sub3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [Ellipsis, 0, Ellipsis])

    def test_dim_mismatch1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], xp.arange(3), [0])

    def test_dim_mismatch2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0], xp.arange(3), [0], [0])

    def test_dim_mismatch3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(6).reshape(2, 3), [0, 0])

    def test_too_many_dims1(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(3, [0])

    def test_too_many_dims2(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(2), [0, 1])

    def test_too_many_dims3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.einsum(xp.arange(6).reshape(2, 3), [Ellipsis, 0, 1, 2])


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

    @testing.for_all_dtypes(no_bool=False)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_unary(self, xp, dtype):
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        out = xp.einsum(self.subscripts, a)
        if xp is not numpy:
            optimized_out = xp.einsum(self.subscripts, a, optimize=True)
            testing.assert_allclose(optimized_out, out)
        return out

    @testing.for_all_dtypes(no_bool=False)
    @testing.numpy_cupy_equal()
    def test_einsum_unary_views(self, xp, dtype):
        a = testing.shaped_arange(self.shape_a, xp, dtype)
        b = xp.einsum(self.subscripts, a)

        return b.ndim == 0 or b.base is a

    @testing.for_all_dtypes_combination(
        ['dtype_a', 'dtype_out'],
        no_bool=False,
        no_complex=True)  # avoid ComplexWarning
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_unary_dtype(self, xp, dtype_a, dtype_out):
        if not numpy.can_cast(dtype_a, dtype_out):
            pytest.skip()
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        return xp.einsum(self.subscripts, a, dtype=dtype_out)


class TestEinSumUnaryOperationWithScalar(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_int(self, xp, dtype):
        return xp.asarray(xp.einsum('->', 2, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_scalar_float(self, xp, dtype):
        return xp.asarray(xp.einsum('', 2.0, dtype=dtype))


@testing.parameterize(*augument_einsum_testcases(
    # dot vecvec
    {'shape_a': (3,), 'shape_b': (3,),
     'subscripts': 'i,i'},
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
        no_bool=False,
        no_float16=False)
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
    {'shape_a': (2, 3, 4), 'shape_b': (4, 3), 'shape_c': (3, 3, 4),
     'subscripts': 'a...,...b,c...->abc...'},
    {'shape_a': (2, 3, 4), 'shape_b': (3, 4), 'shape_c': (3, 3, 4),
     'subscripts': 'a...,...,c...->ac...'},
    {'shape_a': (3, 3, 4), 'shape_b': (4, 3), 'shape_c': (2, 3, 4),
     'subscripts': 'a...,...b,c...->abc...'},
    {'shape_a': (3, 3, 4), 'shape_b': (3, 4), 'shape_c': (2, 3, 4),
     'subscripts': 'a...,...,c...->ac...'},
))
class TestEinSumTernaryOperation(unittest.TestCase):
    @testing.for_all_dtypes_combination(
        ['dtype_a', 'dtype_b', 'dtype_c'],
        no_bool=False,
        no_float16=False)
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum_ternary(self, xp, dtype_a, dtype_b, dtype_c):
        a = testing.shaped_arange(self.shape_a, xp, dtype_a)
        b = testing.shaped_arange(self.shape_b, xp, dtype_b)
        c = testing.shaped_arange(self.shape_c, xp, dtype_c)

        try:
            out = xp.einsum(self.subscripts, a, b, c, optimize=False)
        except TypeError:
            assert xp is numpy
            out = xp.einsum(self.subscripts, a, b, c)

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


@testing.parameterize(*([
    # memory constraint
    {'subscript': 'a,b,c->abc', 'opt': ('greedy', 0)},
    {'subscript': 'acdf,jbje,gihb,hfac', 'opt': ('greedy', 0)},
] + testing.product({'subscript': [
    # long paths
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'chd,bde,agbc,hiad,bdi,cgh,agdb',
    # edge cases
    'eb,cb,fb->cef',
    'dd,fb,be,cdb->cef',
    'bca,cdb,dbf,afc->',
    'dcc,fce,ea,dbf->ab',
    'a,ac,ab,ad,cd,bd,bc->',
], 'opt': ['greedy', 'optimal'],
})))
class TestEinSumLarge(unittest.TestCase):

    def setUp(self):
        chars = 'abcdefghij'
        sizes = numpy.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
        size_dict = {}
        for size, char in zip(sizes, chars):
            size_dict[char] = size

        # Builds views based off initial operands
        string = self.subscript
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(numpy.random.rand(*dims))

        self.operands = operands

    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_einsum(self, xp):
        # TODO(kataoka): support memory efficient cupy.einsum
        with warnings.catch_warnings(record=True) as ws:
            # I hope there's no problem with np.einsum for these cases...
            out = xp.einsum(*self.operands, optimize=self.opt)
            if xp is not numpy and \
                    isinstance(self.opt, tuple):  # with memory limit
                for w in ws:
                    assert 'memory' in str(w.message)
            else:
                assert len(ws) == 0
        return out
