import unittest

import numpy
import pytest

import cupy
from cupy import testing


class ChebTestBase(unittest.TestCase):

    def _get_input(self, xp, in_type, dtype):
        if in_type == 'poly1d':
            return xp.poly1d(testing.shaped_arange((5,), xp, dtype) + 1)
        if in_type == 'ndarray':
            return testing.shaped_arange((5,), xp, dtype)
        if in_type == 'python_scalar':
            return dtype(5).item()
        if in_type == 'numpy_scalar':
            return dtype(5)
        assert False


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['chebadd', 'chebsub', 'chebmul'],
    'type_l': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
    'type_r': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
}))
class TestChebArithmeticBinaryOpTypeCombination(ChebTestBase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_chebarithmetic(self, xp, dtype):
        func = getattr(xp.polynomial.chebyshev, self.fname)
        a = self._get_input(xp, self.type_l, dtype)
        b = self._get_input(xp, self.type_r, dtype)
        return func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['chebadd', 'chebsub', 'chebmul'],
    'shape1': [(), (3,), (5,)],
    'shape2': [(), (3,), (5,)],
}))
class TestChebArithmeticBinaryOpShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_chebarithmetic(self, xp, dtype):
        func = getattr(xp.polynomial.chebyshev, self.fname)
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['chebadd', 'chebsub', 'chebmul'],
    'shape': [(0,), (3, 5)],
}))
class TestChebArithmeticBinaryOpInvalidShape(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    def test_chebarithmetic(self, dtype):
        for xp in (numpy, cupy):
            func = getattr(xp.polynomial.chebyshev, self.fname)
            a = testing.shaped_arange(self.shape, xp, dtype)
            b = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(ValueError):
                func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['chebadd', 'chebsub', 'chebmul'],
}))
class TestChebArithmeticBinaryOpInvalidType(unittest.TestCase):

    def test_chebarithmetic(self):
        for xp in (numpy, cupy):
            func = getattr(xp.polynomial.chebyshev, self.fname)
            a = testing.shaped_arange((5,), xp, bool)
            with pytest.raises(ValueError):
                func(a, a)


@testing.gpu
@testing.parameterize(*testing.product({
    'fname': ['chebadd', 'chebsub', 'chebmul'],
}))
class TestChebyshevArithmeticDiffTypes(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_chebarithmetic_array(self, xp, dtype1, dtype2):
        func = getattr(xp.polynomial.chebyshev, self.fname)
        a = testing.shaped_arange((5,), xp, dtype1)
        b = testing.shaped_arange((5,), xp, dtype2)
        return func(a, b)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_chebarithmetic_poly1d(self, xp, dtype1, dtype2):
        func = getattr(xp.polynomial.chebyshev, self.fname)
        a = xp.poly1d(testing.shaped_arange((5,), xp, dtype1))
        b = xp.poly1d(testing.shaped_arange((5,), xp, dtype2))
        return func(a, b)


@testing.gpu
@testing.parameterize(*testing.product({
    'type': ['poly1d', 'ndarray', 'python_scalar', 'numpy_scalar'],
}))
class TestChebArithmeticUnaryOpTypeCombination(ChebTestBase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_chebmulx(self, xp, dtype):
        a = self._get_input(xp, self.type, dtype)
        return xp.polynomial.chebyshev.chebmulx(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3)
    def test_chebpow(self, xp, dtype):
        a = self._get_input(xp, self.type, dtype)
        return xp.polynomial.chebyshev.chebpow(a, 5)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(0,), (3, 5)],
}))
class TestChebArithmeticUnaryOpInvalidShape(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    def test_chebmulx(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.chebyshev.chebmulx(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_chebpow(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.chebyshev.chebpow(a, 3)


@testing.gpu
class TestChebArithmeticUnaryOpInvalidType(unittest.TestCase):

    def test_chebmulx(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5,), xp, bool)
            with pytest.raises(ValueError):
                xp.polynomial.chebyshev.chebmulx(a)

    def test_chebpow(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5,), xp, bool)
            with pytest.raises(ValueError):
                xp.polynomial.chebyshev.chebpow(a, 5)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(), (1,), (3,), (5,)],
}))
class TestChebmulxShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_chebmulx(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.polynomial.chebyshev.chebmulx(a)


@testing.gpu
@testing.parameterize(*testing.product({
    'shape': [(), (1,), (3,), (5,)],
    'pow': [0, 1, 4, 5],
    'maxpower': [None, 5]
}))
class TestChebpowParametersCombination(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3)
    def test_chebpow(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return xp.polynomial.chebyshev.chebpow(a, self.pow, self.maxpower)


@testing.gpu
@testing.parameterize(*testing.product({
    'pow': [-3, 1.5, 17],
    'maxpower': [5]
}))
class TestChebpowInvalidValue(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    def test_chebpow(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.chebyshev.chebpow(a, self.pow, self.maxpower)


@testing.gpu
@testing.parameterize(*testing.product({
    'type': ['poly1d', 'ndarray'],
}))
class TestChebpowInvalidPowerType(ChebTestBase):

    @testing.for_all_dtypes(no_bool=True)
    def test_chebpow(self, dtype):
        for xp in (numpy, cupy):
            a = self._get_input(xp, self.type, dtype)
            with pytest.raises(TypeError):
                xp.polynomial.chebyshev.chebpow(a, a)
