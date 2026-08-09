from __future__ import annotations

import pytest

from cupy import testing


@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (1,),
    (10,),
    (2, 3),
    (2, 0, 4),
    (2, 3, 4),
    (2, 3, 4, 5),
    (1, 1, 1, 1),
])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('order', ['C', 'F'])
class TestByteswap:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap(self, xp, dtype, shape, inplace, order):
        a = testing.shaped_arange(shape, xp, dtype, order)
        b = a.byteswap(inplace=inplace)
        if inplace:
            assert b is a
        return b


@pytest.mark.parametrize('slices', [
    (slice(None, None, 2),),
    (slice(None, None, 3),),
    (slice(1, None, 2),),
    (slice(None, None, -1),),
])
@pytest.mark.parametrize('inplace', [False, True])
class TestByteswapNonContiguous:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap_strided_1d(self, xp, dtype, slices, inplace):
        a = testing.shaped_arange((12,), xp, dtype)
        b = a[slices].byteswap(inplace=inplace)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap_strided_2d(self, xp, dtype, slices, inplace):
        a = testing.shaped_arange((4, 6), xp, dtype)
        b = a[slices + slices].byteswap(inplace=inplace)
        return b


@pytest.mark.parametrize('inplace', [False, True])
class TestByteswapTransposed:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap_transposed_2d(self, xp, dtype, inplace):
        a = testing.shaped_arange((3, 4), xp, dtype)
        return a.T.byteswap(inplace=inplace)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap_transposed_3d(self, xp, dtype, inplace):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.transpose(2, 0, 1).byteswap(inplace=inplace)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_byteswap_transposed_4d(self, xp, dtype, inplace):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.transpose(3, 1, 0, 2).byteswap(inplace=inplace)
