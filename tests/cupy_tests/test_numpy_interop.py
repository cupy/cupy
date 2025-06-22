from __future__ import annotations

import contextlib
import os
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


class TestAsnumpy:

    def test_asnumpy(self):
        x = testing.shaped_random((2, 3, 4), cupy, cupy.float64)
        y = cupy.asnumpy(x)
        testing.assert_array_equal(x, y)

    def test_asnumpy_out(self):
        x = testing.shaped_random((2, 3, 4), cupy, cupy.float64)
        y = cupyx.empty_like_pinned(x)
        y = cupy.asnumpy(x, out=y)
        testing.assert_array_equal(x, y)
        assert isinstance(y.base, cupy.cuda.PinnedMemoryPointer)
        assert y.base.ptr == y.ctypes.data

    @pytest.mark.skipif(
        int(os.environ.get('CUPY_ENABLE_UMP', 0)) == 1,
        reason='blocking or not is irrelevant when zero-copy is on'
    )
    @pytest.mark.parametrize('blocking', (True, False))
    def test_asnumpy_blocking(self, blocking):
        prefactor = 4
        a = cupy.random.random(prefactor*128*1024*1024, dtype=cupy.float64)
        cupy.cuda.Device().synchronize()

        # Idea: perform D2H copy on a nonblocking stream, during which we try
        # to "corrupt" the host data via NumPy operation. If the copy is
        # properly ordered, corruption would not be possible. Here we craft a
        # problem size and use pinned memory to ensure the failure can be
        # always triggered. (The CUDART API reference ("API synchronization
        # behavior") states that copying between device and pageable memory
        # "might be" synchronous, whereas between device and page-locked
        # memory "should be" fully asynchronous.)
        s = cupy.cuda.Stream(non_blocking=True)
        with s:
            c = cupyx.empty_pinned(a.shape, dtype=a.dtype)
            cupy.asnumpy(a, out=c, blocking=blocking)
            c[c.size//2:] = -1.  # potential data race
        s.synchronize()

        a[c.size//2:] = -1.
        if not blocking:
            ctx = pytest.raises(AssertionError)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            assert cupy.allclose(a, c)
