import unittest

import numpy
import pytest

import cupy
from cupy import testing
import cupyx

try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False


@testing.gpu
class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_1(self):
        n1 = numpy.array([2], numpy.float32)
        c1 = cupy.array([2], numpy.float32)
        csr1 = cupyx.scipy.sparse.csr_matrix((5, 3), dtype=numpy.float32)

        assert numpy is cupy.get_array_module()
        assert numpy is cupy.get_array_module(n1)
        assert cupy is cupy.get_array_module(c1)
        assert cupy is cupy.get_array_module(csr1)

        assert numpy is cupy.get_array_module(n1, n1)
        assert cupy is cupy.get_array_module(c1, c1)
        assert cupy is cupy.get_array_module(csr1, csr1)

        assert cupy is cupy.get_array_module(n1, csr1)
        assert cupy is cupy.get_array_module(csr1, n1)
        assert cupy is cupy.get_array_module(c1, n1)
        assert cupy is cupy.get_array_module(n1, c1)
        assert cupy is cupy.get_array_module(c1, csr1)
        assert cupy is cupy.get_array_module(csr1, c1)

        if scipy_available:
            csrn1 = scipy.sparse.csr_matrix((5, 3), dtype=numpy.float32)

            assert numpy is cupy.get_array_module(csrn1)
            assert cupy is cupy.get_array_module(csrn1, csr1)
            assert cupy is cupy.get_array_module(csr1, csrn1)
            assert cupy is cupy.get_array_module(c1, csrn1)
            assert cupy is cupy.get_array_module(csrn1, c1)
            assert numpy is cupy.get_array_module(n1, csrn1)
            assert numpy is cupy.get_array_module(csrn1, n1)


class MockArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    __array_priority__ = 20  # less than cupy.ndarray.__array_priority__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == '__call__'
        name = ufunc.__name__
        return name, inputs, kwargs


@testing.gpu
class TestArrayUfunc:

    def test_add(self):
        x = cupy.array([3, 7])
        y = MockArray()
        assert x + y == ('add', (x, y), {})
        assert y + x == ('add', (y, x), {})
        y2 = y
        y2 += x
        assert y2 == ('add', (y, x), {'out': y})
        with pytest.raises(TypeError):
            x += y

    @pytest.mark.xfail(
        reason='cupy.ndarray.__array_ufunc__ does not support gufuncs yet')
    def test_matmul(self):
        x = cupy.array([3, 7])
        y = MockArray()
        assert x @ y == ('matmul', (x, y), {})
        assert y @ x == ('matmul', (y, x), {})
        y2 = y
        y2 @= x
        assert y2 == ('matmul', (y, x), {'out': y})
        with pytest.raises(TypeError):
            x @= y

    def test_lt(self):
        x = cupy.array([3, 7])
        y = MockArray()
        assert (x < y) == ('less', (x, y), {})
        assert (y < x) == ('less', (y, x), {})


class MockArray2:
    __array_ufunc__ = None

    def __add__(self, other):
        return 'add'

    def __radd__(self, other):
        return 'radd'

    def __matmul__(self, other):
        return 'matmul'

    def __rmatmul__(self, other):
        return 'rmatmul'

    def __lt__(self, other):
        return 'lt'

    def __gt__(self, other):
        return 'gt'


@testing.gpu
class TestArrayUfuncOptout:

    def test_add(self):
        x = cupy.array([3, 7])
        y = MockArray2()
        assert x + y == 'radd'
        assert y + x == 'add'

    def test_matmul(self):
        x = cupy.array([3, 7])
        y = MockArray2()
        assert x @ y == 'rmatmul'
        assert y @ x == 'matmul'

    def test_lt(self):
        x = cupy.array([3, 7])
        y = MockArray2()
        assert (x < y) == 'gt'
        assert (y < x) == 'lt'
