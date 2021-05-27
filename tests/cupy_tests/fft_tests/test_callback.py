import contextlib
import string
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

import cupy
from cupy import testing


@contextlib.contextmanager
def use_temporary_cache_dir():
    target = 'cupy.fft._callback.get_cache_dir'
    with tempfile.TemporaryDirectory() as path:
        with mock.patch(target, lambda: path):
            yield path


_load_callback = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= 2.5;
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_load_callback_with_aux = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= *((${aux_type}*)callerInfo);
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_load_callback_with_aux2 = r'''
__device__ ${data_type} CB_ConvertInput(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= ((${aux_type}*)callerInfo)[offset];
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = CB_ConvertInput;
'''

_store_callback = r'''
__device__ void CB_ConvertOutput(
    void *dataOut, size_t offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= 3.8;
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = CB_ConvertOutput;
'''

_store_callback_with_aux = r'''
__device__ void CB_ConvertOutput(
    void *dataOut, size_t offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= *((${aux_type}*)callerInfo);
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = CB_ConvertOutput;
'''


def _set_load_cb(code, element, data_type, callback_type, aux_type=None):
    return string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        load_type=callback_type,
        element=element)


def _set_store_cb(code, element, data_type, callback_type, aux_type=None):
    return string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        store_type=callback_type,
        element=element)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10, 7), (10,), (10, 10)],
    'norm': [None, 'ortho'],
}))
@testing.with_requires('cython>=0.29.0')
@testing.gpu
@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='callbacks are only supported on Linux')
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support callbacks')
class Test1dCallbacks:

    def _test_load_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR')
        else:
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
        cb_load = _set_load_cb(code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                if fft_func != 'irfft':
                    out = out.astype(np.complex64)
                else:
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(cb_load=cb_load):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfft')

    def _test_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        code = _store_callback
        if dtype == np.complex64:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                types = ('x', 'cufftReal', 'cufftCallbackStoreR')
        elif dtype == np.complex128:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
            else:
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        cb_store = _set_store_cb(code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(cb_store=cb_store):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfft')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback
        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_store=cb_store):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_aux(self, xp, dtype):
        fft = xp.fft.fft
        c = _load_callback_with_aux2
        if dtype == np.complex64:
            cb_load = _set_load_cb(
                c, 'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
        else:
            cb_load = _set_load_cb(
                c, 'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ', 'double')

        a = testing.shaped_random(self.shape, xp, dtype)
        out_last = self.n if self.n is not None else self.shape[-1]
        out_shape = list(self.shape)
        out_shape[-1] = out_last
        last_min = min(self.shape[-1], out_last)
        b = xp.arange(np.prod(out_shape), dtype=xp.dtype(dtype).char.lower())
        b = b.reshape(out_shape)
        if xp is np:
            x = np.zeros(out_shape, dtype=dtype)
            x[..., 0:last_min] = a[..., 0:last_min]
            x.real *= b
            out = fft(x, n=self.n, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                out = out.astype(np.complex64)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_load_aux_arr=b):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = (
                    'x', 'cufftReal', 'cufftCallbackLoadR', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackLoadD', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        else:  # irfft
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x', 'cufftReal', 'cufftCallbackStoreR', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x', 'cufftDoubleReal',
                               'cufftCallbackStoreD', 'double')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, n=self.n, norm=self.norm)
            if fft_func != 'irfft':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_store=cb_store,
                        cb_load_aux_arr=load_aux, cb_store_aux_arr=store_aux):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifft')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfft')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfft')


@testing.parameterize(
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, None), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': (1, 5), 'axes': None, 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
    {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, None), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': (1, 4, 10), 'axes': None, 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': (-3, -2, -1), 'norm': None},
    {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
    {'shape': (2, 3, 4), 's': (2, 3), 'axes': (0, 1, 2), 'norm': 'ortho'},
)
@testing.with_requires('cython>=0.29.0')
@testing.gpu
@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='callbacks are only supported on Linux')
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support callbacks')
class TestNdCallbacks:

    def _test_load_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR')
        else:
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
        cb_load = _set_load_cb(load_code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if dtype in (np.float32, np.complex64):
                if fft_func != 'irfftn':
                    out = out.astype(np.complex64)
                else:
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(cb_load=cb_load):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfftn')

    def _test_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        store_code = _store_callback
        if dtype == np.complex64:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                types = ('x', 'cufftReal', 'cufftCallbackStoreR')
        elif dtype == np.complex128:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
            else:
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        cb_store = _set_store_cb(store_code, *types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(cb_store=cb_store):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback
        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC')
            else:
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR')
            else:
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_store=cb_store):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = (
                    'x', 'cufftReal', 'cufftCallbackLoadR', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC', 'float')
            else:
                load_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackLoadD', 'double')
                store_types = ('x.y', 'cufftDoubleComplex',
                               'cufftCallbackStoreZ', 'double')
        else:  # irfftn
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC', 'float')
                store_types = (
                    'x', 'cufftReal', 'cufftCallbackStoreR', 'float')
            else:
                load_types = ('x.x', 'cufftDoubleComplex',
                              'cufftCallbackLoadZ', 'double')
                store_types = ('x', 'cufftDoubleReal',
                               'cufftCallbackStoreD', 'double')
        cb_load = _set_load_cb(load_code, *load_types)
        cb_store = _set_store_cb(store_code, *store_types)

        a = testing.shaped_random(self.shape, xp, dtype)
        if xp is np:
            a.real *= 2.5
            out = fft(a, s=self.s, axes=self.axes, norm=self.norm)
            if fft_func != 'irfftn':
                out.imag /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.complex64)
            else:
                out /= 3.8
                if dtype in (np.float32, np.complex64):
                    out = out.astype(np.float32)
        else:
            with use_temporary_cache_dir():
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_store=cb_store,
                        cb_load_aux_arr=load_aux, cb_store_aux_arr=store_aux):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifftn')

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfftn')

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfftn')
