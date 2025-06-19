from __future__ import annotations


import os

from cupy_builder._context import (
    _get_env_bool, _get_env_path, Context)


class TestGetEnvBool:
    def test_true(self) -> None:
        assert _get_env_bool('V',  {}, default=True)
        assert _get_env_bool('V', {'V': '1'}, default=True)
        assert _get_env_bool('V', {'V': '1'})
        assert _get_env_bool('V', {'V': '1'}, default=False)
        # inappropriate usage
        assert _get_env_bool('V', {'V': 'True'})
        assert _get_env_bool('V', {'V': 'False'})

    def test_false(self) -> None:
        assert not _get_env_bool('V', {})
        assert not _get_env_bool('V', {}, default=False)
        assert not _get_env_bool('V', {'V': '0'})
        assert not _get_env_bool('V', {'V': '0'}, default=False)
        assert not _get_env_bool('V', {'V': '0'}, default=True)


class TestGetEnvPath:
    def test(self) -> None:
        assert _get_env_path('P', {}) == []
        assert _get_env_path('P', {'P': f'1{os.pathsep}'}) == ['1']
        assert _get_env_path('P', {'P': f'1{os.pathsep}2'}) == ['1', '2']


class TestContext:
    def test_default(self) -> None:
        ctx = Context('.', _env={}, _argv=[])

        assert ctx.source_root == '.'
        assert ctx.setup_command == ''
        assert not ctx.use_cuda_python
        assert not ctx.use_hip
        assert not ctx.use_stub
        assert ctx.include_dirs == []
        assert ctx.library_dirs == []
        assert ctx.long_description_path is None
        assert ctx.wheel_metadata_path is None
        assert not ctx.profile
        assert not ctx.linetrace
        assert not ctx.annotate
        assert not ctx.no_rpath
        assert ctx.win32_cl_exe_path is None

        # Deprecated
        assert ctx.wheel_libs == []
        assert ctx.wheel_includes == []

    def test_env(self) -> None:
        ctx = Context('.', _env={
            'CUPY_USE_CUDA_PYTHON': '1',
            'CUPY_INSTALL_USE_HIP': '1',
            'CUPY_INCLUDE_PATH': f'/tmp/include{os.pathsep}/tmp2/include',
            'CUPY_LIBRARY_PATH': f'/tmp/lib{os.pathsep}/tmp2/lib',
            'CUPY_INSTALL_LONG_DESCRIPTION': 'foo.rst',
            'CUPY_INSTALL_WHEEL_METADATA': '_wheel.json',
            'CUPY_INSTALL_NO_RPATH': '1',
            'CUPY_INSTALL_CYTHON_PROFILE': '1',
            'CUPY_INSTALL_CYTHON_COVERAGE': '1',
            'CUPY_INSTALL_NO_MEANING': '1',  # dummy
        }, _argv=[])

        assert ctx.use_cuda_python
        assert ctx.use_hip
        assert ctx.include_dirs == ['/tmp/include', '/tmp2/include']
        assert ctx.library_dirs == ['/tmp/lib', '/tmp2/lib']
        assert ctx.long_description_path == 'foo.rst'
        assert ctx.wheel_metadata_path == '_wheel.json'
        assert ctx.no_rpath
        assert ctx.profile
        assert ctx.annotate
        assert ctx.linetrace
