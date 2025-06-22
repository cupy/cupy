from __future__ import annotations

import unittest
import sys

import numpy
import pytest

import cupy
from cupy._core import core
from cupy import testing
from cupy_tests.core_tests import test_raw


class TestSize(unittest.TestCase):

    def tearDown(self):
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size_axis(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a, axis=1)

    @testing.for_all_dtypes()
    def test_size_axis_error(self, dtype):
        for xp in (numpy, cupy):
            a = xp.ndarray((2, 3), dtype=dtype)
            with pytest.raises(IndexError):
                return xp.size(a, axis=3)

    @testing.numpy_cupy_equal()
    @testing.slow
    def test_size_huge(self, xp):
        a = xp.ndarray(2 ** 32, 'b')  # 4 GiB
        return xp.size(a)


_orders = {
    order_arg: order_expect
    for order_expect, order_args in [
        ('C', ['C', 'c', 'CONTIGUOUS', '', None]),
        ('F', ['F', 'f', 'FORTRAN']),
    ]
    for order_arg in order_args
}


class TestOrder(unittest.TestCase):

    @testing.for_orders(_orders.keys())
    def test_ndarray(self, order):
        order_expect = _orders[order]
        a = core.ndarray((2, 3), order=order)
        expect_c = order_expect == 'C'
        expect_f = order_expect == 'F'
        assert a.flags.c_contiguous == expect_c
        assert a.flags.f_contiguous == expect_f


class TestMinScalarType:
    def test_scalar(self):
        for v in (-129, -128, 0, 1.2, numpy.inf):
            assert cupy.min_scalar_type(v) is numpy.min_scalar_type(v)

    @testing.for_all_dtypes()
    def test_numpy_scalar(self, dtype):
        sc = dtype(1)
        for v in (sc, [sc, sc]):
            assert cupy.min_scalar_type(v) is numpy.min_scalar_type(v)

    @testing.for_all_dtypes()
    def test_cupy_scalar(self, dtype):
        sc = cupy.array(-1).astype(dtype)
        for v in (sc, [sc, sc]):
            assert cupy.min_scalar_type(v) is sc.dtype

    @testing.for_all_dtypes()
    def test_numpy_ndarray(self, dtype):
        arr = numpy.array([[-1, 1]]).astype(dtype)
        for v in (arr, (arr, arr)):
            assert cupy.min_scalar_type(v) is numpy.min_scalar_type(v)

    @testing.for_all_dtypes()
    def test_cupy_ndarray(self, dtype):
        arr = cupy.array([[-1, 1]]).astype(dtype)
        for v in (arr, (arr, arr)):
            assert cupy.min_scalar_type(v) is arr.dtype


@testing.parameterize(*testing.product({
    'cxx': (None, '--std=c++11'),
}))
class TestCuPyHeaders(unittest.TestCase):

    def setUp(self):
        self.temporary_cache_dir_context = test_raw.use_temporary_cache_dir()
        self.cache_dir = self.temporary_cache_dir_context.__enter__()
        self.header = '\n'.join(['#include <' + h + '>'
                                 for h in core._cupy_header_list])

    def tearDown(self):
        self.temporary_cache_dir_context.__exit__(*sys.exc_info())

    def test_compiling_core_header(self):
        code = r'''
        extern "C" __global__ void _test_ker_() { }
        '''
        code = self.header + code
        options = () if self.cxx is None else (self.cxx,)
        ker = cupy.RawKernel(code, '_test_ker_',
                             options=options, backend='nvrtc')
        ker((1,), (1,), ())
        cupy.cuda.Device().synchronize()
