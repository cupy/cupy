import numpy
import pytest

import cupy

from cupy import testing
from cupy._core._gufuncs import _GUFunc


class TestGUFuncOrder():

    @pytest.mark.parametrize("order", ['C', 'F', 'K'])
    @testing.numpy_cupy_array_equal(strides_check=True)
    def test_order(self, xp, order):
        x = testing.shaped_arange((2, 3, 4), xp=xp)
        if xp is cupy:
            def default(x):
                return x
            gu_func = _GUFunc(default, '(i, j, k)->(i, j, k)')
            return gu_func(x, order=order)
        else:
            return xp.asarray(x, order=order)

    @pytest.mark.parametrize("order", [('F', 'C', 'C'), ('F', 'F', 'F')])
    def test_order_a(self, order):
        x = testing.shaped_arange((2, 3, 4), order=order[0])
        y = testing.shaped_arange((2, 3, 4), order=order[1])

        def default(x, y):
            return x

        gu_func = _GUFunc(default, '(i,j,k),(i,j,k)->(i,j,k)')
        z = gu_func(x, y, order='A')
        if order[2] == 'C':
            assert z.flags.c_contiguous
        else:
            assert z.flags.f_contiguous


class TestGUFuncSignatures():
    def test_signatures(self):
        dtypes = 'fdihq'
        dtypes_access = {d: None for d in dtypes}

        def integers(x, y):
            nonlocal dtypes_access
            dtypes_access[numpy.dtype(x.dtype).char] = integers
            return x + y

        def floats(x, y):
            nonlocal dtypes_access
            dtypes_access[numpy.dtype(x.dtype).char] = floats
            return x + y

        def default(x, y):
            nonlocal dtypes_access
            dtypes_access[numpy.dtype(x.dtype).char] = default
            return x + y

        sigs = (('ii->i', integers), ('dd->d', floats))
        gu_func = _GUFunc(default, '(i),(i)->(i)', signatures=sigs)
        for dtype in dtypes:
            x = cupy.array([10], dtype=dtype)
            y = x
            gu_func(x, y, casting='no')
            if dtype in 'i':
                assert dtypes_access[dtype] == integers
            elif dtype in 'd':
                assert dtypes_access[dtype] == floats
            else:
                assert dtypes_access[dtype] == default

    @pytest.mark.parametrize("sig,", ['ii->i', 'i', ('i', 'i', 'i')])
    def test_signature_lookup(self, sig):
        called = False

        def func(x, y):
            nonlocal called
            called = True
            return x + y

        def default(x, y):
            return x + y

        dtypes = 'fdhq'

        sigs = (('ii->i', func),)
        gu_func = _GUFunc(default, '(i),(i)->(i)', signatures=sigs)
        for dtype in dtypes:
            x = cupy.array([10], dtype=dtype)
            y = x
            gu_func(x, y, casting='no')
            assert not called

        x = cupy.array([10], dtype='d')
        y = x
        z = gu_func(x, y, casting='unsafe', signature=sig)
        assert z.dtype == numpy.int32
        assert called

    @pytest.mark.parametrize("sigs,", [('i',), ('',), ('iii->i',), ('ii->',)])
    def test_invalid_signatures(self, sigs):

        def default(x, y):
            return x + y

        with pytest.raises(ValueError):
            _GUFunc(default, '(i),(i)->(i)', signatures=sigs)

    @pytest.mark.parametrize("sig,", ['i->i', 'id->i', ''])
    def test_invalid_lookup(self, sig):

        def default(x, y):
            return x + y

        sigs = (('ii->i', default),)
        gu_func = _GUFunc(default, '(i),(i)->(i)', signatures=sigs)
        _GUFunc(default, '(i),(i)->(i)', signatures=sigs)

        x = cupy.array([10], dtype='d')
        y = x
        with pytest.raises(TypeError):
            gu_func(x, y, casting='unsafe', signature=sig)
