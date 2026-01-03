from __future__ import annotations

import numpy
import pytest

import cupy

from cupy import testing
from cupy._core._gufuncs import _GUFunc


class TestGUFuncSignature:
    @pytest.mark.parametrize('signature', [
        ('(i,j)->(i,j)',
         [(('i', False, False), ('j', False, False))],
         [(('i', False, False), ('j', False, False))], 2),
        ('->(i)', [()], [(('i', False, False),)], 1),
        ('(i,j),(j,k)->(k,l)',
         [(('i', False, False), ('j', False, False)),
          (('j', False, False), ('k', False, False))],
         [(('k', False, False), ('l', False, False))], 4),
        ('()->()', [()], [()], 0),
        ('(i?,j|1),(i?,j)->(i?,j)',
         [(('i', True, False), ('j', False, True)),
          (('i', True, False), ('j', False, False))],
         [(('i', True, False), ('j', False, False))], 2),
    ])
    def test_signature_parsing(self, signature):
        i, o, n_cd = cupy._core._gufuncs._parse_gufunc_signature(signature[0])
        assert i == signature[1]
        assert o == signature[2]
        assert n_cd == signature[3]

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

    def _get_gufunc_scalar_supports_all(self, signature):
        def func(x, out=None):
            # Does not use keepdims, but gufunc supports it.
            return x.sum(axis=-1, out=out)
        return _GUFunc(
            func, signature, supports_batched=True, supports_out=True)

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

    @pytest.mark.parametrize('axes', [
        [(0, 1), (0, 1), (0, 1)],
        [(0, 1), (0, 1), (1, 0)],
        [(-2, -1), (-3, 0), (-1, -3)],
    ])
    @pytest.mark.parametrize('use_out', [True, False])
    @testing.numpy_cupy_array_equal()
    def test_axes_matmul(self, xp, axes, use_out):
        # Do not use a weird shape, but rather rely on each
        # arange transpose giving a unique result.
        x = testing.shaped_arange((3, 3, 3, 3), xp=xp)
        y = testing.shaped_arange((3, 3, 3, 3), xp=xp)
        if use_out:
            out = xp.empty((3, 3, 3, 3))
        else:
            out = None

        return xp.matmul(x, y, axes=axes, out=out)

    @pytest.mark.parametrize('ax,outer_ax',
                             [(0, 1), (1, 0), ((-1,), 0)])
    @testing.numpy_cupy_array_equal(accept_error=numpy.exceptions.AxisError)
    def test_axes_single_matmul(self, xp, ax, outer_ax):
        # We do not allow this (just as NumPy), although it may be possible
        # to define it in principle.
        x = xp.ones((2, 3))
        y = xp.ones((2, 3))
        xp.matmul(x, y, axes=[ax] * 2 + [()])
        # no return, should raise error.

    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @pytest.mark.parametrize('keepdims', [True, False])
    @testing.numpy_cupy_array_equal()
    def test_axis(self, xp, axis, keepdims):
        x = testing.shaped_arange((2, 3, 4, 5), xp=xp)
        if xp is cupy:
            return self._get_gufunc_scalar('(i)->()')(
                x, axis=axis, keepdims=keepdims)
        else:
            return x.sum(axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @pytest.mark.parametrize('keepdims', [True, False])
    @testing.numpy_cupy_array_equal()
    def test_axis_full_core_support(self, xp, axis, keepdims):
        x = testing.shaped_arange((2, 3, 4, 5), xp=xp)
        if xp is cupy:
            return self._get_gufunc_scalar_supports_all('(i)->()')(
                x, axis=axis, keepdims=keepdims)
        else:
            return x.sum(axis=axis, keepdims=keepdims)

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


class TestGUFuncOrder:

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


class TestGUFuncSignatures:
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


class TestGUFuncOptional:
    def _get_gufunc_ridiculous_optional(self):
        signature = '(a?,b,c,d?),(i?,j?,k,l)->(b,c,a?,d?,k,l,j?,i?)'

        def func(x, y):
            # The ufunc is always passed all dimensions (filled in with 1)
            # if omitted and optional.
            res_shape = x.shape[1:-1] + (x.shape[0], x.shape[-1])
            res_shape += y.shape[2:] + (y.shape[1], y.shape[0])
            return cupy.ones(res_shape)

        return _GUFunc(func, signature)

    def _get_forbidden_optional(self):
        signature = '(a?,b?),(b,a?)->(a?,b?)'

        def func(x, y):
            raise RuntimeError('this will not be called')

        return _GUFunc(func, signature)

    @pytest.mark.parametrize('x_ndim, y_ndim', [
        (2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4), (4, 3), (3, 4),
        (4, 4), (6, 6)
    ])
    def test_ridiculous_optional(self, x_ndim, y_ndim):
        gufunc = self._get_gufunc_ridiculous_optional()

        x_shape = tuple(range(1, x_ndim + 1))
        y_shape = tuple(range(1, y_ndim + 1))
        x = cupy.ones(x_shape)
        y = cupy.ones(y_shape)
        # Succeeds if the correct `func` above matches with allocated output.
        res = gufunc(x, y)

        if x_ndim == 6 and y_ndim == 6:
            # only test where this is the case
            x_shape = x_shape[2:]
            y_shape = y_shape[2:]
            outer_shape = (1, 2)
        else:
            outer_shape = ()

        # Check that the result shape is actually what we expect it to be.
        if x.ndim == 2:  # b, c
            core_shape = x_shape
        elif x.ndim == 3:  # b, c, d -> b, c, d
            core_shape = x_shape[:-1] + (x_shape[-1],)
        else:  # a, b, c, d -> b, c, a, d
            core_shape = x_shape[1:-1] + (x_shape[0], x_shape[-1])

        if y.ndim == 2:  # k, l
            core_shape += y_shape
        elif y.ndim == 3:  # j, k, l -> k, l, j
            core_shape += y_shape[1:] + (y_shape[0],)
        else:  # i, j, k, l -> k, l, j, i
            core_shape += y_shape[2:] + (y_shape[1], y_shape[0])

        assert res.shape == outer_shape + core_shape

    def test_forbidden_optional(self):
        gufunc = self._get_forbidden_optional()
        x = cupy.ones(2)
        y = cupy.ones((2, 2))
        with pytest.raises(ValueError):
            # first op is missing a at front but second is not
            gufunc(x, y)

        with pytest.raises(ValueError):
            # second op is missing a at end but first is not
            gufunc(y, x)


class TestGUFuncBroadcastable:
    def _get_gufunc(self):
        def func(x, y):
            shape = cupy.broadcast_shapes(x.shape, y.shape)
            return cupy.ones(shape)
        return _GUFunc(func, '(i|1,j|1),(i|1,j)->(i,j)')

    @pytest.mark.parametrize('x_shape, y_shape', [
        ((2, 1), (2, 3)),
        ((1, 1), (2, 1)),
        ((2, 3), (1, 3)),
        ((1, 1), (1, 1)),
    ])
    def test_broadcastable(self, x_shape, y_shape):
        func = self._get_gufunc()
        x = cupy.ones(x_shape)
        y = cupy.ones(y_shape)

        res = func(x, y)
        assert res.shape == cupy.broadcast_shapes(x_shape, y_shape)

    @pytest.mark.parametrize('x_shape, y_shape', [
        ((2, 3), (2, 1)),  # second operand 1 is not broadcastable
    ])
    def test_not_broadcastable(self, x_shape, y_shape):
        func = self._get_gufunc()
        x = cupy.ones(x_shape)
        y = cupy.ones(y_shape)

        with pytest.raises(ValueError):
            func(x, y)
