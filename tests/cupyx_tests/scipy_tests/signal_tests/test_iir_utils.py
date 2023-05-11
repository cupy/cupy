import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing
from cupyx.scipy.signal._iir_utils import apply_iir


@pytest.mark.xfail(
    runtime.is_hip and driver.get_build_version() < 5_00_00000,
    reason='name_expressions with ROCm 4.3 may not work')
@testing.with_requires('scipy')
class TestIIRUtils:
    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=5e-5)
    def test_order(self, size, order, in_dtype, const_dtype, xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x_scale = 0.5 if xp.dtype(in_dtype).kind not in {'i', 'u'} else 1

        x = testing.shaped_random((size,), xp, in_dtype, scale=x_scale)
        a = testing.shaped_random((order,), xp, dtype=const_dtype, scale=1)
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
        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.5)
    def test_order_ndim(self, size, order, axis, in_dtype, const_dtype,
                        xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x_scale = 0.5 if xp.dtype(in_dtype).kind not in {'i', 'u'} else 1
        c_scale = 0.2 if xp.dtype(const_dtype).kind not in {'i', 'u'} else 1

        x = testing.shaped_random((4, 5, 3, size), xp, in_dtype, scale=x_scale)
        a = testing.shaped_random(
            (order,), xp, dtype=const_dtype, scale=c_scale)
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
        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_order_zero_starting(self, size, order, in_dtype, const_dtype,
                                 xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x_scale = 0.5 if xp.dtype(in_dtype).kind not in {'i', 'u'} else 1
        c_scale = 0.2 if xp.dtype(const_dtype).kind not in {'i', 'u'} else 1

        x = testing.shaped_random((size,), xp, in_dtype, scale=x_scale)
        a = testing.shaped_random(
            (order,), xp, dtype=const_dtype, scale=c_scale)
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

        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res

    @pytest.mark.parametrize('size', [11, 32, 51, 64, 100])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=5e-2)
    def test_order_zero_starting_ndim(
            self, size, order, axis, in_dtype, const_dtype, xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x = testing.shaped_random((3, 2, 3, size), xp, in_dtype, scale=1)
        a = testing.shaped_random((order,), xp, dtype=const_dtype, scale=2)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)

        zi_shape = list(x.shape)
        zi_shape[axis] = order
        zi = xp.zeros(zi_shape, dtype=in_dtype)

        res = None
        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], axis, zi, dtype=out_dtype, block_sz=32)
        else:
            b = xp.ones([1], dtype=const_dtype)
            zi = xp.moveaxis(zi, axis, -1)
            zi_m_shape = zi.shape
            zi = zi.reshape(-1, order).copy()
            zi = xp.concatenate([scp.signal.lfiltic(b, a, z) for z in zi])
            zi = zi.reshape(zi_m_shape[:-1] + (-1,))
            zi = xp.moveaxis(zi, -1, axis)
            res, _ = scp.signal.lfilter(xp.ones(1, dtype=const_dtype),
                                        a, x, zi=zi, axis=axis)
        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res

    @pytest.mark.parametrize('size', [11, 20, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=5)
    def test_order_starting_cond(
            self, size, order, in_dtype, const_dtype, xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x_scale = 0.5 if xp.dtype(in_dtype).kind not in {'i', 'u'} else 1
        c_scale = 0.2 if xp.dtype(const_dtype).kind not in {'i', 'u'} else 1

        x = testing.shaped_random((size,), xp, in_dtype, scale=x_scale)
        a = testing.shaped_random(
            (order,), xp, dtype=const_dtype, scale=c_scale)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)
        zi = testing.shaped_random((order,), xp, dtype=in_dtype)
        res = None
        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], zi=zi, dtype=out_dtype, block_sz=32)
        else:
            b = xp.ones([1], dtype=const_dtype)
            zi = scp.signal.lfiltic(b, a, zi[::-1])
            res, _ = scp.signal.lfilter(b, a, x, zi=zi)

        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res

    @pytest.mark.parametrize('size', [11, 32, 51, 64, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3])
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    @testing.for_all_dtypes_combination(
        no_float16=True, no_bool=True, names=('in_dtype', 'const_dtype'))
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=5)
    def test_order_starting_cond_ndim(
            self, size, order, axis, in_dtype, const_dtype, xp, scp):
        out_dtype = xp.result_type(in_dtype, const_dtype)
        if xp.dtype(out_dtype).kind in {'i', 'u'}:
            pytest.skip()

        x_scale = 0.5 if xp.dtype(in_dtype).kind not in {'i', 'u'} else 1
        c_scale = 0.2 if xp.dtype(const_dtype).kind not in {'i', 'u'} else 1

        x = testing.shaped_random((3, 2, 3, size), xp, in_dtype, scale=x_scale)
        a = testing.shaped_random(
            (order,), xp, dtype=const_dtype, scale=c_scale)
        a = xp.r_[1, a]
        a = a.astype(const_dtype)

        zi_shape = list(x.shape)
        zi_shape[axis] = order
        zi = testing.shaped_random(zi_shape, xp, dtype=in_dtype)

        res = None
        if xp is cupy:
            const_dtype = xp.dtype(const_dtype)
            if const_dtype.kind == 'u':
                const_dtype = xp.dtype(const_dtype.char.lower())
                a = a.astype(const_dtype)
            res = apply_iir(x, -a[1:], axis, zi, dtype=out_dtype, block_sz=32)
        else:
            b = xp.ones([1], dtype=const_dtype)
            zi = xp.moveaxis(zi, axis, -1)
            zi_m_shape = zi.shape
            zi = zi.reshape(-1, order).copy()
            zi = xp.concatenate([
                scp.signal.lfiltic(b, a, z[::-1]) for z in zi])
            zi = zi.reshape(zi_m_shape[:-1] + (-1,))
            zi = xp.moveaxis(zi, -1, axis)
            res, _ = scp.signal.lfilter(xp.ones(1, dtype=const_dtype),
                                        a, x, zi=zi, axis=axis)
        res = xp.nan_to_num(res, nan=xp.nan, posinf=xp.nan, neginf=xp.nan)
        return res
