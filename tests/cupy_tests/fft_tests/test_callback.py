from __future__ import annotations

import contextlib
import os
import string
import sys
import tempfile
from unittest import mock

try:
    import Cython
except ImportError:
    Cython = None
else:
    if Cython.__version__ < '0.29.0':
        Cython = None
import numpy as np
import pytest

import cupy
from cupy import testing
from cupy.cuda import cufft
from cupy.cuda.device import get_compute_capability


def cuda_version():
    return cupy.cuda.runtime.runtimeGetVersion()


cb_ver_for_test = ('legacy', 'jit')


def check_should_skip_legacy_test():
    if not sys.platform.startswith('linux'):
        pytest.skip('legacy callbacks are only supported on Linux')
    if Cython is None:
        pytest.skip("no working Cython")
    if 'LD_PRELOAD' in os.environ:
        pytest.skip("legacy callback does not work if libcufft.so "
                    "is preloaded")
    if cufft.getVersion() == 12000 and get_compute_capability() == '75':
        pytest.skip('cuFFT legacy callbacks in CUDA 13.0.0 do not support '
                    'cc 7.5')
    if cufft.getVersion() == 11303 and get_compute_capability() == '120':
        pytest.skip('cuFFT legacy callbacks in CUDA 12.8.0 do not support '
                    'cc 12.0')


def check_should_skip_jit_test():
    if cufft.getVersion() < 11303:
        pytest.skip('JIT callbacks require cuFFT from CUDA 12.8+')


@contextlib.contextmanager
def use_temporary_cache_dir():
    target = 'cupy.fft._callback.get_cache_dir'
    with tempfile.TemporaryDirectory() as path:
        with mock.patch(target, lambda: path):
            yield path


suppress_legacy_warning = pytest.mark.filterwarnings(
    "ignore:.*legacy callback.*:DeprecationWarning")


_load_callback = r'''
__device__ ${data_type} ${cb_name}(
    void* dataIn, ${offset_type} offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= 2.5;
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = ${cb_name};
'''

_load_callback_with_aux = r'''
__device__ ${data_type} ${cb_name}(
    void* dataIn, ${offset_type} offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= *((${aux_type}*)callerInfo);
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = ${cb_name};
'''

_load_callback_with_aux2 = r'''
__device__ ${data_type} ${cb_name}(
    void* dataIn, ${offset_type} offset, void* callerInfo, void* sharedPtr)
{
    ${data_type} x = ((${data_type}*)dataIn)[offset];
    ${element} *= ((${aux_type}*)callerInfo)[offset];
    return x;
}

__device__ ${load_type} d_loadCallbackPtr = ${cb_name};
'''

_store_callback = r'''
__device__ void ${cb_name}(
    void *dataOut, ${offset_type} offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= 3.8;
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = ${cb_name};
'''

_store_callback_with_aux = r'''
__device__ void ${cb_name}(
    void *dataOut, ${offset_type} offset, ${data_type} element,
    void *callerInfo, void *sharedPointer)
{
    ${data_type} x = element;
    ${element} /= *((${aux_type}*)callerInfo);
    ((${data_type}*)dataOut)[offset] = x;
}

__device__ ${store_type} d_storeCallbackPtr = ${cb_name};
'''


def _set_load_cb(
        code, element, data_type, callback_type, callback_name,
        aux_type=None, cb_ver=''):
    if cb_ver == 'jit':
        callback_type = callback_type.replace(
            'cufftCallback', 'cufftJITCallback')
    callback = string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        load_type=callback_type,
        cb_name=callback_name,
        element=element,
        offset_type=('size_t' if cb_ver == 'legacy' else 'unsigned long long'))
    if cb_ver == 'jit':
        callback = "#include <cufftXt.h>\n\n" + callback
    return callback


def _set_store_cb(
        code, element, data_type, callback_type, callback_name,
        aux_type=None, cb_ver=''):
    if cb_ver == 'jit':
        callback_type = callback_type.replace(
            'cufftCallback', 'cufftJITCallback')
    callback = string.Template(code).substitute(
        data_type=data_type,
        aux_type=aux_type,
        store_type=callback_type,
        cb_name=callback_name,
        element=element,
        offset_type=('size_t' if cb_ver == 'legacy' else 'unsigned long long'))
    if cb_ver == 'jit':
        callback = "#include <cufftXt.h>\n\n" + callback
    return callback


# Note: this class is place here instead of at the end of this file, because
# pytest does not reset warnings internally, and other tests would suppress
# the warnings such that at the end we have no warnings to capture, but we want
# to ensure warnings are raised.
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support callbacks')
class TestInputValidationWith1dCallbacks:

    shape = (10,)
    norm = 'ortho'
    dtype = np.complex64

    def test_fft_load_legacy(self):
        check_should_skip_legacy_test()

        fft = cupy.fft.fft
        code = _load_callback
        types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                 'cufftJITCallbackLoadComplex')
        cb_load = _set_load_cb(code, *types, cb_ver='legacy')

        a = testing.shaped_random(self.shape, cupy, self.dtype)
        with pytest.deprecated_call(
                match='legacy callback is considered deprecated'):
            with use_temporary_cache_dir():
                with cupy.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_ver='legacy'):
                    fft(a, norm=self.norm)

    def test_fft_load_jit_no_name(self):
        check_should_skip_jit_test()

        fft = cupy.fft.fft
        code = _load_callback
        types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                 'cufftJITCallbackLoadComplex')
        cb_load = _set_load_cb(code, *types, cb_ver='jit')

        a = testing.shaped_random(self.shape, cupy, self.dtype)
        with use_temporary_cache_dir():
            # We omit passing cb_load_name. The test infra setup would check
            # if we can infer it correctly.
            with cupy.fft.config.set_cufft_callbacks(
                    cb_load=cb_load, cb_ver='jit'):
                fft(a, norm=self.norm)

    def test_fft_store_legacy(self):
        check_should_skip_legacy_test()

        fft = cupy.fft.fft
        code = _store_callback
        types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                 'cufftJITCallbackStoreComplex')
        cb_store = _set_store_cb(code, *types, cb_ver='legacy')

        a = testing.shaped_random(self.shape, cupy, self.dtype)
        with pytest.deprecated_call(
                match='legacy callback is considered deprecated'):
            with use_temporary_cache_dir():
                with cupy.fft.config.set_cufft_callbacks(
                        cb_store=cb_store, cb_ver='legacy'):
                    fft(a, norm=self.norm)

    def test_fft_store_jit_no_name(self):
        check_should_skip_jit_test()

        fft = cupy.fft.fft
        code = _store_callback
        types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                 'cufftJITCallbackStoreComplex')
        cb_store = _set_store_cb(code, *types, cb_ver='jit')

        a = testing.shaped_random(self.shape, cupy, self.dtype)
        with use_temporary_cache_dir():
            # We omit passing cb_store_name. The test infra setup would check
            # if we can infer it correctly.
            with cupy.fft.config.set_cufft_callbacks(
                    cb_store=cb_store, cb_ver='jit'):
                fft(a, norm=self.norm)

    def test_fft_load_store_legacy_aux(self):
        check_should_skip_legacy_test()

        fft = cupy.fft.fft
        dtype = self.dtype
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        load_aux = cupy.asarray(2.5, dtype=cupy.dtype(dtype).char.lower())
        store_aux = cupy.asarray(3.8, dtype=cupy.dtype(dtype).char.lower())

        load_types = (
            'x.x', 'cufftComplex', 'cufftCallbackLoadC',
            'cufftJITCallbackLoadComplex', 'float')
        store_types = (
            'x.y', 'cufftComplex', 'cufftCallbackStoreC',
            'cufftJITCallbackStoreComplex', 'float')
        cb_load = _set_load_cb(load_code, *load_types, cb_ver='legacy')
        cb_store = _set_store_cb(store_code, *store_types, cb_ver='legacy')

        a = testing.shaped_random(self.shape, cupy, self.dtype)
        with pytest.deprecated_call(
                match='cb_load_aux_arr or cb_store_aux_arr is deprecated'), \
            pytest.deprecated_call(
                match='legacy callback is considered deprecated'):
            with use_temporary_cache_dir():
                with cupy.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_store=cb_store,
                        cb_load_aux_arr=load_aux, cb_store_aux_arr=store_aux,
                        cb_ver='legacy'):
                    fft(a, norm=self.norm)


@testing.parameterize(*testing.product({
    'n': [None, 5, 10, 15],
    'shape': [(10, 7), (10,), (10, 10)],
    'norm': [None, 'ortho'],
    'cb_ver': cb_ver_for_test,
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support callbacks')
class Test1dCallbacks:

    def _test_load_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        # for simplicity we use the JIT callback names for both legacy/jit
        fft = getattr(xp.fft, fft_func)
        code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                     'cufftJITCallbackLoadComplex')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                     'cufftJITCallbackLoadDoubleComplex')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR',
                     'cufftJITCallbackLoadReal')
        else:  # float64
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                     'cufftJITCallbackLoadDoubleReal')
        cb_load = _set_load_cb(code, *types, cb_ver=self.cb_ver)
        cb_load_name = types[-1] if self.cb_ver == 'jit' else None

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
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifft')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfft')

    def _test_store_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = getattr(xp.fft, fft_func)
        code = _store_callback

        # for simplicity we use the JIT callback names for both legacy/jit
        if dtype == np.complex64:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                         'cufftJITCallbackStoreComplex')
            else:  # float32 for irfft
                types = ('x', 'cufftReal', 'cufftCallbackStoreR',
                         'cufftJITCallbackStoreReal')
        elif dtype == np.complex128:
            if fft_func != 'irfft':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                         'cufftJITCallbackStoreDoubleComplex')
            else:  # float64 for irfft
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                         'cufftJITCallbackStoreDoubleReal')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                     'cufftJITCallbackStoreComplex')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                     'cufftJITCallbackStoreDoubleComplex')
        cb_store = _set_store_cb(code, *types, cb_ver=self.cb_ver)
        cb_store_name = types[-1] if self.cb_ver == 'jit' else None

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
                with xp.fft.config.set_cufft_callbacks(
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifft')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfft')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        # for simplicity we use the JIT callback names for both legacy/jit
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback
        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR',
                              'cufftJITCallbackLoadReal')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex')
            else:  # float64
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                              'cufftJITCallbackLoadDoubleReal')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR',
                               'cufftJITCallbackStoreReal')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                               'cufftJITCallbackStoreDoubleReal')
        cb_load = _set_load_cb(load_code, *load_types, cb_ver=self.cb_ver)
        cb_load_name = load_types[-1] if self.cb_ver == 'jit' else None
        cb_store = _set_store_cb(store_code, *store_types, cb_ver=self.cb_ver)
        cb_store_name = store_types[-1] if self.cb_ver == 'jit' else None

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
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifft')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_aux(self, xp, dtype):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = xp.fft.fft
        c = _load_callback_with_aux2
        # for simplicity we use the JIT callback names for both legacy/jit
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                     'cufftJITCallbackLoadComplex', 'float')
        else:  # complex128
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                     'cufftJITCallbackLoadDoubleComplex', 'double')
        cb_load = _set_load_cb(c, *types, cb_ver=self.cb_ver)
        cb_load_name = types[3] if self.cb_ver == 'jit' else None

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
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_load_data=b.data, cb_ver=self.cb_ver):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        # for simplicity we use the JIT callback names for both legacy/jit
        if fft_func in ('fft', 'ifft'):
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC',
                    'cufftJITCallbackLoadComplex', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC',
                    'cufftJITCallbackStoreComplex', 'float')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex', 'double')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex', 'double')
        elif fft_func == 'rfft':
            if dtype == np.float32:
                load_types = (
                    'x', 'cufftReal', 'cufftCallbackLoadR',
                    'cufftJITCallbackLoadReal', 'float')
                store_types = (
                    'x.y', 'cufftComplex', 'cufftCallbackStoreC',
                    'cufftJITCallbackStoreComplex', 'float')
            else:  # float64
                load_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                    'cufftJITCallbackLoadDoubleReal', 'double')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex', 'double')
        else:  # irfft
            if dtype == np.complex64:
                load_types = (
                    'x.x', 'cufftComplex', 'cufftCallbackLoadC',
                    'cufftJITCallbackLoadComplex', 'float')
                store_types = (
                    'x', 'cufftReal', 'cufftCallbackStoreR',
                    'cufftJITCallbackStoreReal', 'float')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex', 'double')
                store_types = (
                    'x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                    'cufftJITCallbackStoreDoubleReal', 'double')
        cb_load = _set_load_cb(load_code, *load_types, cb_ver=self.cb_ver)
        cb_load_name = load_types[3] if self.cb_ver == 'jit' else None
        cb_store = _set_store_cb(store_code, *store_types, cb_ver=self.cb_ver)
        cb_store_name = store_types[3] if self.cb_ver == 'jit' else None

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
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_load_data=load_aux.data,
                        cb_store_data=store_aux.data,
                        cb_ver=self.cb_ver):
                    out = fft(a, n=self.n, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_fft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_ifft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifft')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_rfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfft')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, contiguous_check=False)
    def test_irfft_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfft')


@testing.parameterize(*(
    testing.product_dict(
        [
            {'shape': (3, 4), 's': None, 'axes': None, 'norm': None},
            {'shape': (3, 4), 's': (1, 5), 'axes': (-2, -1), 'norm': None},
            {'shape': (3, 4), 's': None, 'axes': (-2, -1), 'norm': None},
            {'shape': (3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
            {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': None},
            {'shape': (2, 3, 4), 's': (1, 4, 10),
             'axes': (-3, -2, -1), 'norm': None},
            {'shape': (2, 3, 4), 's': None,
             'axes': (-3, -2, -1), 'norm': None},
            {'shape': (2, 3, 4), 's': None, 'axes': None, 'norm': 'ortho'},
            {'shape': (2, 3, 4), 's': (2, 3), 'axes': (
                0, 1, 2), 'norm': 'ortho'},
        ],

        testing.product(
            {'cb_ver': cb_ver_for_test, },
        ),
    )
))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='hipFFT does not support callbacks')
class TestNdCallbacks:

    def _test_load_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        # for simplicity we use the JIT callback names for both legacy/jit
        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        if dtype == np.complex64:
            types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                     'cufftJITCallbackLoadComplex')
        elif dtype == np.complex128:
            types = ('x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                     'cufftJITCallbackLoadDoubleComplex')
        elif dtype == np.float32:
            types = ('x', 'cufftReal', 'cufftCallbackLoadR',
                     'cufftJITCallbackLoadReal')
        else:  # float64
            types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                     'cufftJITCallbackLoadDoubleReal')
        cb_load = _set_load_cb(load_code, *types, cb_ver=self.cb_ver)
        cb_load_name = types[3] if self.cb_ver == 'jit' else None

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
                with xp.fft.config.set_cufft_callbacks(
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'fftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'ifftn')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'rfftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load(self, xp, dtype):
        return self._test_load_helper(xp, dtype, 'irfftn')

    def _test_store_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = getattr(xp.fft, fft_func)
        store_code = _store_callback

        # for simplicity we use the JIT callback names for both legacy/jit
        if dtype == np.complex64:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                         'cufftJITCallbackStoreComplex')
            else:  # float32 for irfftn
                types = ('x', 'cufftReal', 'cufftCallbackStoreR',
                         'cufftJITCallbackStoreReal')
        elif dtype == np.complex128:
            if fft_func != 'irfftn':
                types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                         'cufftJITCallbackStoreDoubleComplex')
            else:  # float64 for irfftn
                types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                         'cufftJITCallbackStoreDoubleReal')
        elif dtype == np.float32:
            types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                     'cufftJITCallbackStoreComplex')
        elif dtype == np.float64:
            types = ('x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                     'cufftJITCallbackStoreDoubleComplex')
        cb_store = _set_store_cb(store_code, *types, cb_ver=self.cb_ver)
        cb_store_name = types[3] if self.cb_ver == 'jit' else None

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
                with xp.fft.config.set_cufft_callbacks(
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'fftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'ifftn')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'rfftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_store(self, xp, dtype):
        return self._test_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback
        store_code = _store_callback

        # for simplicity we use the JIT callback names for both legacy/jit
        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR',
                              'cufftJITCallbackLoadReal')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex')
            else:  # float64
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                              'cufftJITCallbackLoadDoubleReal')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex')
        else:  # irfft
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR',
                               'cufftJITCallbackStoreReal')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                               'cufftJITCallbackStoreDoubleReal')
        cb_load = _set_load_cb(load_code, *load_types, cb_ver=self.cb_ver)
        cb_load_name = load_types[3] if self.cb_ver == 'jit' else None
        cb_store = _set_store_cb(store_code, *store_types, cb_ver=self.cb_ver)
        cb_store_name = store_types[3] if self.cb_ver == 'jit' else None

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
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_ver=self.cb_ver):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'fftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'ifftn')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'rfftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store(self, xp, dtype):
        return self._test_load_store_helper(xp, dtype, 'irfftn')

    def _test_load_store_aux_helper(self, xp, dtype, fft_func):
        if self.cb_ver == 'legacy':
            check_should_skip_legacy_test()
        else:
            check_should_skip_jit_test()

        fft = getattr(xp.fft, fft_func)
        load_code = _load_callback_with_aux
        store_code = _store_callback_with_aux
        if xp is cupy:
            load_aux = xp.asarray(2.5, dtype=xp.dtype(dtype).char.lower())
            store_aux = xp.asarray(3.8, dtype=xp.dtype(dtype).char.lower())

        # for simplicity we use the JIT callback names for both legacy/jit
        if fft_func in ('fftn', 'ifftn'):
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex', 'float')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex', 'float')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex', 'double')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex', 'double')
        elif fft_func == 'rfftn':
            if dtype == np.float32:
                load_types = ('x', 'cufftReal', 'cufftCallbackLoadR',
                              'cufftJITCallbackLoadReal', 'float')
                store_types = ('x.y', 'cufftComplex', 'cufftCallbackStoreC',
                               'cufftJITCallbackStoreComplex', 'float')
            else:  # float64
                load_types = ('x', 'cufftDoubleReal', 'cufftCallbackLoadD',
                              'cufftJITCallbackLoadDoubleReal', 'double')
                store_types = (
                    'x.y', 'cufftDoubleComplex', 'cufftCallbackStoreZ',
                    'cufftJITCallbackStoreDoubleComplex', 'double')
        else:  # irfftn
            if dtype == np.complex64:
                load_types = ('x.x', 'cufftComplex', 'cufftCallbackLoadC',
                              'cufftJITCallbackLoadComplex', 'float')
                store_types = ('x', 'cufftReal', 'cufftCallbackStoreR',
                               'cufftJITCallbackStoreReal', 'float')
            else:  # complex128
                load_types = (
                    'x.x', 'cufftDoubleComplex', 'cufftCallbackLoadZ',
                    'cufftJITCallbackLoadDoubleComplex', 'double')
                store_types = ('x', 'cufftDoubleReal', 'cufftCallbackStoreD',
                               'cufftJITCallbackStoreDoubleReal', 'double')
        cb_load = _set_load_cb(load_code, *load_types, cb_ver=self.cb_ver)
        cb_load_name = load_types[3] if self.cb_ver == 'jit' else None
        cb_store = _set_store_cb(store_code, *store_types, cb_ver=self.cb_ver)
        cb_store_name = store_types[3] if self.cb_ver == 'jit' else None

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
                        cb_load=cb_load, cb_load_name=cb_load_name,
                        cb_store=cb_store, cb_store_name=cb_store_name,
                        cb_load_data=load_aux.data,
                        cb_store_data=store_aux.data,
                        cb_ver=self.cb_ver):
                    out = fft(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_fftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'fftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_ifftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'ifftn')

    @suppress_legacy_warning
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_rfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'rfftn')

    @suppress_legacy_warning
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4, atol=1e-7, accept_error=ValueError,
                                 contiguous_check=False)
    def test_irfftn_load_store_aux(self, xp, dtype):
        return self._test_load_store_aux_helper(xp, dtype, 'irfftn')
