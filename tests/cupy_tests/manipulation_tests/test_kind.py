import unittest
import pytest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestKind(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_asfortranarray1(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous
            return ret.strides
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray2(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous
            return ret.strides
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray3(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3, 4), dtype)
            ret = xp.asfortranarray(xp.asfortranarray(x))
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous
            return ret.strides
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray4(self, dtype):
        def func(xp):
            x = xp.zeros((2, 3), dtype)
            x = xp.transpose(x, (1, 0))
            ret = xp.asfortranarray(x)
            assert ret.flags.f_contiguous
            return ret.strides
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_asfortranarray5(self, dtype):
        def func(xp):
            x = testing.shaped_arange((2, 3), xp, dtype)
            ret = xp.asfortranarray(x)
            assert x.flags.c_contiguous
            assert ret.flags.f_contiguous
            return ret.strides
        assert func(numpy) == func(cupy)

    @testing.for_all_dtypes()
    def test_require_flag_check(self, dtype):
        possible_flags = [['C_CONTIGUOUS'], ['F_CONTIGUOUS']]
        x = cupy.zeros((2, 3, 4), dtype)
        for flags in possible_flags:
            arr = cupy.require(x, dtype, flags)
            for parameter in flags:
                assert arr.flags[parameter]
                assert arr.dtype == dtype

    @testing.for_all_dtypes()
    def test_require_owndata(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype)
        arr = x.view()
        arr = cupy.require(arr, dtype, ['O'])
        assert arr.flags['OWNDATA']

    @testing.for_all_dtypes()
    def test_require_C_and_F_flags(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype)
        with pytest.raises(ValueError):
            cupy.require(x, dtype, ['C', 'F'])

    @testing.for_all_dtypes()
    def test_require_incorrect_requirments(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype)
        with pytest.raises(ValueError):
            cupy.require(x, dtype, ['W'])

    @testing.for_all_dtypes()
    def test_require_incorrect_dtype(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype)
        with pytest.raises(ValueError):
            cupy.require(x, 'random', 'C')

    @testing.for_all_dtypes()
    def test_require_empty_requirements(self, dtype):
        x = cupy.zeros((2, 3, 4), dtype)
        x = cupy.require(x, dtype, [])
        assert x.flags['C_CONTIGUOUS']
