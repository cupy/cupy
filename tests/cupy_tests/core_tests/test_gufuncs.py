import numpy
import pytest

import cupy

from cupy import testing
from cupy._core._gufuncs import _GUFunc


class TestGUFuncSignature:
    @pytest.mark.parametrize('signature', [
        ('(i,j)->(i,j)', [('i', 'j')], [('i', 'j')]),
        ('->(i)', [()], [('i',)]),
        ('(i,j),(j,k)->(k,l)', [('i', 'j'), ('j', 'k')], [('k', 'l')]),
        ('()->()', [()], [()])])
    def test_signature_parsing(self, signature):
        i, o = cupy._core._gufuncs._parse_gufunc_signature(signature[0])
        assert i == signature[1]
        assert o == signature[2]

    @pytest.mark.parametrize('signature', [
        '(i,j)(i,j)',
        '(i,j)-(i,j)',
        '(i,j)(i,j)->(i,j)',
        'j->(i',
        '',
        '()->()->'])
    def test_invalid_signature_parsing(self, signature):
        with pytest.raises(ValueError):
            cupy._core._gufuncs._parse_gufunc_signature(signature)


class TestGUFuncAxes:
    def _get_gufunc(self, signature):
        def func(x):
            return x
        return _GUFunc(func, signature)

    def _get_gufunc_scalar(self, signature):
        def func(x):
            return x.sum()
        return _GUFunc(func, signature)

    @pytest.mark.parametrize('axes', [
        ((-1, -2), (-1, -2)),
        ((0, 1), (0, 1)),
        ((0, 1), (-1, -2)),
        ((1, 2), (-1, -2)),
        ((1, 2), (1, 2)),
        ((1, 2), (2, 3)),
        ((2, 3), (-1, -2)),
        ((2, 3), (0, 1)),
        ((2, 3), (1, 2)),
        ((0, 3), (1, 2)),
        ((0, 3), (2, 0)),
    ])
    @testing.numpy_cupy_array_equal()
    def test_axes_selection(self, xp, axes):
        x = testing.shaped_arange((2, 3, 4, 5), xp=xp)
        if xp is cupy:
            return self._get_gufunc('(i,j)->(i,j)')(x, axes=list(axes))
        else:
            return numpy.moveaxis(x, axes[0], axes[1])

    @pytest.mark.parametrize('axes', [
        (-1, -2),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 2),
        (0, 3),
        (1, 3),
        (3, 0),
        (2, 0),
        (2, 1),
        (1, 0),
    ])
    @testing.numpy_cupy_array_equal()
    def test_axes_selection_single(self, xp, axes):
        x = testing.shaped_arange((2, 3, 4, 5), xp=xp)
        if xp is cupy:
            return self._get_gufunc('(i)->(i)')(x, axes=list(axes))
        else:
            return numpy.moveaxis(x, axes[0], axes[1])

    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.numpy_cupy_array_equal()
    def test_axis(self, xp, axis):
        x = testing.shaped_arange((2, 3, 4, 5), xp=xp)
        if xp is cupy:
            return self._get_gufunc_scalar('(i)->()')(x, axis=axis)
        else:
            return x.sum(axis=axis)

    def test_axis_invalid(self):
        x = testing.shaped_arange((2, 3, 4, 5))
        with pytest.raises(ValueError):
            self._get_gufunc('(i, j)->(i, j)')(x, axis=((0, 1), (0, 1)))

    @pytest.mark.parametrize('supports_batched', [True, False])
    def test_supports_batched(self, supports_batched):
        x = testing.shaped_arange((2, 3, 4, 5))

        def func(x):
            nonlocal supports_batched
            if supports_batched:
                assert x.ndim == 4
            else:
                assert x.ndim == 2
            return x
        gu_func = _GUFunc(func, '(i,j)->(i,j)',
                          supports_batched=supports_batched)
        gu_func(x)


class TestGUFuncOut:
    def _get_gufunc(self):
        def func(x):
            return x
        return _GUFunc(func, '(i,j)->(i,j)')

    def test_out_array(self):
        x = testing.shaped_arange((2, 3, 4, 5))
        out = cupy.empty((2, 3, 4, 5))
        self._get_gufunc()(x, out=out)
        testing.assert_allclose(x, out)

    def test_supports_out(self):
        x = testing.shaped_arange((2, 3, 4, 5))
        out = cupy.empty((2, 3, 4, 5))
        out_ptr = out.data.ptr

        def func(x, out=None):
            nonlocal out_ptr
            # Base is a view of the output due to the batching
            assert out.base.data.ptr == out_ptr
            out[:] = x
        gu_func = _GUFunc(func, '(i,j)->(i,j)', supports_out=True)
        gu_func(x, out=out)
        testing.assert_allclose(x, out)

    def test_invalid_output_shape(self):
        x = testing.shaped_arange((2, 3, 4, 5))
        out = cupy.empty((3, 3, 4, 5))
        with pytest.raises(ValueError):
            self._get_gufunc()(x, out=out)

    def test_invalid_output_dtype(self):
        x = testing.shaped_arange((2, 3, 4, 5))
        out = cupy.empty((2, 3, 4, 5), dtype='h')
        with pytest.raises(TypeError):
            self._get_gufunc()(x, out=out)


class TestGUFuncDtype:
    @testing.for_all_dtypes(name='dtype_i', no_bool=True, no_complex=True)
    @testing.for_all_dtypes(name='dtype_o', no_bool=True, no_complex=True)
    def test_dtypes(self, dtype_i, dtype_o):
        x = testing.shaped_arange((2, 3, 4, 5), dtype=dtype_i)
        if numpy.can_cast(dtype_o, x.dtype):
            def func(x):
                return x
            gufunc = _GUFunc(func, '(i,j)->(i,j)')
            z = gufunc(x, dtype=dtype_o)
            assert z.dtype == dtype_o
            testing.assert_allclose(z, x)


class TestGUFuncOrder():

    @pytest.mark.parametrize('order', ['C', 'F', 'K'])
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

    @pytest.mark.parametrize('order', [('F', 'C', 'C'), ('F', 'F', 'F')])
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

    @pytest.mark.parametrize('sig,', ['ii->i', 'i', ('i', 'i', 'i')])
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

    @pytest.mark.parametrize('sigs,', [('i',), ('',), ('iii->i',), ('ii->',)])
    def test_invalid_signatures(self, sigs):

        def default(x, y):
            return x + y

        with pytest.raises(ValueError):
            _GUFunc(default, '(i),(i)->(i)', signatures=sigs)

    @pytest.mark.parametrize('sig,', ['i->i', 'id->i', ''])
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
