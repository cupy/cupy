
import pytest

import cupy
from cupy import testing
from cupyx.scipy.signal._iir_utils import apply_iir

import numpy as np


TOL = {
    cupy.float16: 1e-2,
    cupy.float32: 1e-6,
}


@testing.with_requires('scipy')
class TestIIRUtils:
    def _sequential_impl(self, x, b, zi=None):
        t = x.copy()
        out_dtype = cupy.result_type(x.dtype, b.dtype)
        y = cupy.empty_like(x, dtype=out_dtype)

        n = t.size
        k = b.size

        for i in range(n):
            y[i] = t[i]
            upper_bound = i if i < k and zi is None else k
            for j in range(1, upper_bound + 1):
                arr = y
                if i < k and zi is not None:
                    if j > i:
                        arr = zi

                y[i] += b[j - 1] * arr[i - j]
        return y

    def _sequential_impl_nd(self, x, b, axis, zi=None):
        n = x.shape[axis]
        out_dtype = cupy.result_type(x.dtype, b.dtype)

        x = cupy.moveaxis(x, axis, -1)
        x_shape = x.shape
        x = x.reshape(-1, n)

        if zi is not None:
            zi = cupy.moveaxis(zi, axis, -1)
            zi_shape = zi.shape
            zi = zi.reshape(-1, zi_shape[-1])
            zi = zi.copy()

        y = cupy.empty_like(x, dtype=out_dtype)

        for i in range(x.shape[0]):
            # y[i] = self._sequential_impl(x[i], b)
            zi_i = None
            if zi is not None:
                zi_i = zi[i]
            y[i] = apply_iir(x[i], b, zi=zi_i)

        y = y.reshape(x_shape)
        y = cupy.moveaxis(y, -1, axis)
        return y

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    @testing.for_all_dtypes_combination(
        no_complex=True, no_float16=True, no_bool=True,
        names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=2e-5)
    def test_order(self, size, order, in_dtype, const_dtype, xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x = testing.shaped_random((size,), xp, in_dtype)
        a = testing.shaped_random((order,), xp, dtype=const_dtype)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)

        res = None
        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], dtype=out_dtype, block_sz=32)
        else:
            res = scp.signal.lfilter(xp.ones(1, dtype=const_dtype), a, x)
        res = xp.where(xp.isinf(res), xp.nan, res)
        return res

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.for_all_dtypes_combination(
        no_complex=True, no_float16=True, no_bool=True,
        names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.5)
    def test_order_ndim(self, size, order, axis, in_dtype, const_dtype,
                        xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x = testing.shaped_random((4, 5, 3, size), xp, in_dtype)
        a = testing.shaped_random((order,), xp, dtype=const_dtype)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)

        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], axis, dtype=out_dtype, block_sz=32)
        else:
            res = scp.signal.lfilter(xp.ones(1, dtype=const_dtype),
                                     a, x, axis=axis)
        res = xp.where(xp.isinf(res), xp.nan, res)
        return res

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    @testing.for_all_dtypes_combination(
        no_complex=True, no_float16=True, no_bool=True,
        names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-4)
    def test_order_zero_starting(self, size, order, in_dtype, const_dtype,
                                 xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x = testing.shaped_random((size,), xp, in_dtype)
        a = testing.shaped_random((order,), xp, dtype=const_dtype)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)
        zi = xp.zeros(order, dtype=in_dtype)

        res = None
        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], zi=zi, dtype=out_dtype, block_sz=32)
        else:
            b = xp.ones([1], dtype=const_dtype)
            zi = scp.signal.lfiltic(b, a, zi)
            res, _ = scp.signal.lfilter(b, a, x, zi=zi)

        res = xp.where(xp.isinf(res), xp.nan, res)
        return res

    @pytest.mark.parametrize('size', [11, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    def test_order_zero_starting_ndim(self, size, order, axis):
        signs = cupy.tile(cupy.asarray([1, -1], dtype=cupy.float32),
                          int(2 * np.ceil(size / 4)))

        x = [cupy.arange(3 + i, size + 3 + i, dtype=cupy.float32) *
             signs[:size] for i in range(3)]
        x = [cupy.expand_dims(e, 0) for e in x]
        x = cupy.concatenate(x, axis=0)

        final_shape = [3, 2]
        final_shape += list(x.shape)
        zi_shape = list(final_shape)
        zi_shape[axis] = order

        zi = cupy.zeros(zi_shape)

        x = cupy.broadcast_to(x, final_shape)
        x = x.copy()
        b = testing.shaped_random((order,))
        par = apply_iir(x, b, axis, zi)
        seq = self._sequential_impl_nd(x, b, axis, zi)
        testing.assert_allclose(seq, par)

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    def test_order_starting_cond(self, size, order):
        signs = cupy.tile(cupy.asarray([1, -1], dtype=cupy.float32),
                          int(2 * np.ceil(size / 4)))
        zi = testing.shaped_random((order,))
        x = cupy.arange(3, size + 3, dtype=cupy.float32) * signs[:size]
        b = testing.shaped_random((order,))
        seq = self._sequential_impl(x, b, zi=zi)
        par = apply_iir(x, b, zi=zi)
        testing.assert_allclose(seq, par, 1e-6)

    @pytest.mark.parametrize('size', [11, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    def test_order_starting_cond_ndim(self, size, order, axis):
        signs = cupy.tile(cupy.asarray([1, -1], dtype=cupy.float32),
                          int(2 * np.ceil(size / 4)))

        x = [cupy.arange(3 + i, size + 3 + i, dtype=cupy.float32) *
             signs[:size] for i in range(3)]
        x = [cupy.expand_dims(e, 0) for e in x]
        x = cupy.concatenate(x, axis=0)

        final_shape = [3, 2]
        final_shape += list(x.shape)
        zi_shape = list(final_shape)
        zi_shape[axis] = order

        zi = testing.shaped_random(tuple(zi_shape))

        x = cupy.broadcast_to(x, final_shape)
        x = x.copy()
        b = testing.shaped_random((order,))
        par = apply_iir(x, b, axis, zi)
        seq = self._sequential_impl_nd(x, b, axis, zi)
        testing.assert_allclose(seq, par)
