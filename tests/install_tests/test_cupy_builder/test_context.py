from __future__ import annotations


import json
import os
from pathlib import Path
import runpy
import shutil

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
    def test_header_cache(self) -> None:
        source_root = str(Path(__file__).resolve().parents[3])
        ctx = Context(source_root, _env={}, _argv=[])
        # Also validates every include in CuPy-owned repository headers.
        ctx.generate_header_cache()
        cache = json.loads((Path(source_root)
                            / 'cupy/_core/_header_cache.json').read_text())
        assert cache['keys']['full'] == ctx.header_cache_key
        assert cache['headers']['cupy/float16.cuh'] == 'float16'
        assert cache['headers']['cuda/std/type_traits'] == 'common'
        assert cache['headers']['cupy/xsf/config.h'] == 'xsf'

    def test_generated_metadata(self, tmp_path) -> None:
        core = tmp_path / 'cupy' / '_core'
        core.mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[3]
                        / 'cupy/_core/_header_cache.py',
                        core / '_header_cache.py')
        include = core / 'include'
        include.mkdir()
        ctx = Context(str(tmp_path), _env={}, _argv=[])
        ctx.generate_header_cache()
        metadata = core / '_header_cache.json'
        content = metadata.read_text()
        module = runpy.run_path(str(core / '_header_cache.py'))
        assert (module['load_header_cache']()
                == module['build_header_cache'](include))
        before = metadata.stat().st_mtime_ns
        ctx.generate_header_cache()
        assert metadata.stat().st_mtime_ns == before
        (include / 'test.h').write_text('// changed\n')
        ctx.generate_header_cache()
        assert metadata.read_text() != content

    def test_default(self) -> None:
        ctx = Context('.', _env={}, _argv=[])

        assert ctx.source_root == '.'
        assert ctx.setup_command == ''
        assert not ctx.use_cuda_python
        assert not ctx.use_hip
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
