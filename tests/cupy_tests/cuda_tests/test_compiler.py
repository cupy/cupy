from __future__ import annotations

import copy
import pickle
import unittest
from unittest import mock

import pytest

import cupy
from cupy.cuda import compiler


def cuda_version():
    # These NVRTC tests must gate on what compiler._get_arch() /
    # compile_using_nvrtc actually key on -- the NVRTC version -- not
    # runtimeGetVersion(), which reports the *build-time* cudart. The two
    # coincide for a source build but differ when a prebuilt wheel is tested
    # on a CUDA image of a different minor version.
    if cupy.cuda.runtime.is_hip:
        return 0
    major, minor = compiler._get_nvrtc_version()
    return major * 1000 + minor * 10


@unittest.skipIf(cupy.cuda.runtime.is_hip, 'CUDA specific tests')
@pytest.mark.thread_unsafe(reason="Uses mock.patch for (for most tests).")
class TestNvrtcArch(unittest.TestCase):
    def setUp(self):
        cupy.clear_memo()  # _get_arch result is cached

    def _check_get_arch(self, device_cc, expected_arch):
        with mock.patch('cupy.cuda.device.Device') as device_class:
            device_class.return_value.compute_capability = device_cc
            assert compiler._get_arch() == expected_arch
        cupy.clear_memo()  # _get_arch result is cached

    @unittest.skipUnless(
        9000 <= cuda_version() < 13000, 'Requires CUDA 9.x-12.x')
    def test_get_arch_cuda9(self):
        self._check_get_arch('62', '62')  # Tegra
        self._check_get_arch('70', '70')
        self._check_get_arch('72', '72')  # Tegra

    @unittest.skipUnless(10010 <= cuda_version(),
                         'Requires CUDA 10.1 or later')
    def test_get_arch_cuda101(self):
        self._check_get_arch('75', '75')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_get_arch_cuda11(self):
        self._check_get_arch('80', '80')

    @unittest.skipUnless(12080 <= cuda_version(),
                         'Requires CUDA 12.8 or later')
    def test_get_arch_cuda128(self):
        self._check_get_arch('100', '100')
        self._check_get_arch('120', '120')

    def _compile(self, arch):
        compiler.compile_using_nvrtc('', arch=arch)

    @unittest.skipUnless(
        9000 <= cuda_version() < 13000, 'Requires CUDA 9.x-12.x')
    def test_compile_cuda9(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('70')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '73')

    @unittest.skipUnless(10010 <= cuda_version() < 11000,
                         'Requires CUDA 10.1 or 10.2')
    def test_compile_cuda101(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('75')

        # It should fail. (compute_80 is not supported until CUDA 11)
        self.assertRaises(
            compiler.CompileException, self._compile, '80')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_compile_cuda11(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        self._compile('80')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '83')


class TestNvrtcStderr(unittest.TestCase):

    @unittest.skipIf(cupy.cuda.runtime.is_hip,
                     'HIPRTC has different error message')
    def test1(self):
        # An error message contains the file name `kern.cu`
        with self.assertRaisesRegex(compiler.CompileException, 'kern.cu'):
            compiler.compile_using_nvrtc('a')

    @unittest.skipIf(not cupy.cuda.runtime.is_hip,
                     'NVRTC has different error message')
    def test2(self):
        with self.assertRaises(compiler.CompileException) as e:
            compiler.compile_using_nvrtc('a')
            assert "unknown type name 'a'" in e


class TestIsValidKernelName(unittest.TestCase):

    def test_valid(self):
        assert compiler.is_valid_kernel_name('valid_name_1')

    def test_empty(self):
        assert not compiler.is_valid_kernel_name('')

    def test_start_with_digit(self):
        assert not compiler.is_valid_kernel_name('0_invalid')

    def test_new_line(self):
        assert not compiler.is_valid_kernel_name('invalid\nname')

    def test_symbol(self):
        assert not compiler.is_valid_kernel_name('invalid$name')

    def test_space(self):
        assert not compiler.is_valid_kernel_name('invalid name')


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = compiler.CompileException('msg', 'fn.cu', 'fn', ('-ftz=true',))
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


class TestCompileWithCache:
    def test_compile_module_with_cache(self):
        compiler._compile_module_with_cache('__device__ void func() {}')

    @pytest.mark.thread_unsafe(reason='Mocks compiler entry point')
    @pytest.mark.parametrize('method', [
        'compile_with_cache', 'compile_to_ltoir',
    ])
    def test_include_paths_added_later(self, method):
        from cupy._core import core

        options = ('-I/user/headers',)
        with (
            mock.patch.object(compiler, '_compile_module_with_cache') as call,
            mock.patch.object(core, '_get_header_source',
                              side_effect=AssertionError('Unexpected read')),
        ):
            call.return_value = None
            getattr(core, method)('', options=options)
        assert call.call_args.args[1] == options + ('--std=c++17',)
        assert call.call_args.kwargs['cupy_include_dirs']
        assert call.call_args.kwargs['runtime_include_dirs']


@pytest.mark.thread_unsafe(reason='Patches HIP compiler state')
@pytest.mark.parametrize('version,needs_headers', [
    (305, True), (40400000, False),
])
def test_hip_headers_loaded_lazily(version, needs_headers, monkeypatch):
    from cupy._core import core

    getter = mock.Mock(return_value='#pragma once\nstruct CachedHeader {};\n')
    monkeypatch.setattr(core, '_get_header_source', getter)
    monkeypatch.setattr(compiler, '_cuda_hip_version', version)
    monkeypatch.setattr(compiler, '_hip_extra_source', None)
    for source in ('// first kernel', '// second kernel'):
        converted = compiler._convert_to_hip_source(source, is_hiprtc=True)
        assert ('struct CachedHeader' in converted) == needs_headers
        assert source in converted
    assert getter.call_count == int(needs_headers)


@pytest.mark.thread_unsafe(reason='Patches compiler cache state')
class TestHeaderCache:
    @pytest.fixture
    def cache_backend(self, tmp_path, monkeypatch):
        from cupy._core import core
        from cupy.cuda._compiler_cache import DiskKernelCacheBackend

        cache = copy.deepcopy(core.HEADER_CACHE)
        backend = DiskKernelCacheBackend(str(tmp_path / 'cache'))
        save = mock.Mock(wraps=backend.save)
        monkeypatch.setattr(backend, 'save', save)
        monkeypatch.setattr(compiler, '_kernel_cache_backend', backend)
        monkeypatch.setattr(core, 'HEADER_CACHE', cache)
        monkeypatch.setenv('CUPY_CACHE_IN_MEMORY', '0')
        return cache, save

    @pytest.mark.parametrize('bundled', [False, True])
    def test_injected_options_and_cache_paths(self, bundled, monkeypatch):
        from cupy._core import core

        # A cached LTO payload exercises the compiler's key construction and
        # option injection without invoking NVRTC or requiring a GPU.
        backend = mock.Mock()
        backend.load.return_value = b'cached LTO'
        preprocess = mock.Mock(return_value=b'compiler version')
        monkeypatch.setattr(compiler, '_kernel_cache_backend', backend)
        monkeypatch.setattr(compiler, '_preprocess', preprocess)
        monkeypatch.setattr(compiler, '_empty_file_preprocess_cache', {})
        monkeypatch.setattr(compiler, '_get_nvrtc_version', lambda: (13, 0))
        monkeypatch.setattr(compiler, '_get_arch_for_options_for_nvrtc',
                            lambda arch: ('-arch=compute_80', 'ptx'))
        paths = {'bundled': '/first/cupy', 'runtime': '/cuda/include'}
        monkeypatch.setattr(core, 'add_cupy_include_dirs',
                            lambda opts: (f'-I{paths["bundled"]}',) + opts)
        monkeypatch.setattr(core, 'add_runtime_include_dirs',
                            lambda opts: opts + (f'-I{paths["runtime"]}',))

        def compile():
            return compiler._compile_with_cache_cuda(
                '', ('--std=c++17',), '80', to_ltoir=True,
                cupy_include_dirs=bundled, runtime_include_dirs=True)

        assert compile() == b'cached LTO'
        injected = preprocess.call_args.args[1]
        assert '-I/cuda/include' in injected
        assert ('-I/first/cupy' in injected) == bundled
        paths['bundled'] = '/second/cupy'
        compile()
        assert backend.load.call_args_list[0] == backend.load.call_args_list[1]
        paths['runtime'] = '/other/cuda/include'
        compile()
        assert backend.load.call_args_list[1] != backend.load.call_args_list[2]

    @pytest.mark.parametrize('header,group', [
        ('cupy/float16.cuh', 'float16'),
        ('cupy/xsf/config.h', 'xsf'),
    ])
    def test_selective_invalidation(self, header, group, cache_backend):
        from cupy._core import core

        cache, save = cache_backend
        source = (f'#include <{header}>\n'
                  'extern "C" __global__ void header_cache_test() {}')

        core.compile_with_cache(source)
        assert save.call_count == 1
        # Updating an unused group must preserve the cached binary.
        cache['keys']['build'] = 'changed unused group'
        cache['keys']['full'] = 'changed full key'
        core.compile_with_cache(source)
        assert save.call_count == 1
        cache['keys'][group] = 'changed selected group'
        core.compile_with_cache(source)
        assert save.call_count == 2

    @pytest.mark.parametrize('custom_include_path', [False, True])
    def test_external_wrapper_invalidation(
            self, tmp_path, custom_include_path, cache_backend,
            allow_full_header_cache):
        from cupy._core import core

        cache, save = cache_backend
        wrapper = tmp_path / 'wrapper.h'
        wrapper.write_text('#include <cupy/xsf/config.h>\n')
        # Both absolute and search-path includes must recognize the unknown
        # header and select the full key.
        header = 'wrapper.h' if custom_include_path else wrapper.as_posix()
        source = (f'#include "{header}"\n'
                  'extern "C" __global__ void header_cache_test() {}')
        options = (f'-I{tmp_path}',) if custom_include_path else ()
        core.compile_with_cache(source, options)
        core.compile_with_cache(source, options)
        assert save.call_count == 1
        cache['keys']['full'] = 'changed transitive xsf dependency'
        core.compile_with_cache(source, options)
        assert save.call_count == 2

    def test_strict_unknown_header(self):
        from cupy._core import _header_cache

        # conftest enables strict checking throughout the suite.
        with pytest.raises(_header_cache._UnknownHeaderError,
                           match='user_wrapper.h'):
            compiler._compile_module_with_cache(
                '#include "user_wrapper.h"', arch='80')
