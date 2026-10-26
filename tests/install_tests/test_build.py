from __future__ import annotations

from distutils import ccompiler
from distutils import sysconfig
import os
import unittest

import pytest

from cupy_builder._context import Context
from cupy_builder import install_build as build


test_hip = bool(int(os.environ.get('CUPY_INSTALL_USE_HIP', '0')))


@pytest.mark.parametrize(
    ('machine', 'expected'),
    [('AMD64', 'x64'), ('x86_64', 'x64'), ('ARM64', 'arm64'),
     ('aarch64', 'arm64')])
def test_get_win32_cuda_arch(monkeypatch, machine, expected):
    monkeypatch.setattr(build.platform, 'machine', lambda: machine)
    assert build._get_win32_cuda_arch() == expected


class TestDetectCudaMajor:

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        monkeypatch.delenv('CONDA_OVERRIDE_CUDA', raising=False)
        monkeypatch.delenv('CONDA_BUILD_CROSS_COMPILATION', raising=False)
        build.is_conda_cross_compiling.cache_clear()
        yield
        build.is_conda_cross_compiling.cache_clear()

    @staticmethod
    def _write_cuda_h(include_dir, body):
        include_dir.mkdir(parents=True, exist_ok=True)
        (include_dir / 'cuda.h').write_text(body)

    @pytest.mark.parametrize(('cuda_version', 'expected'), [
        (12000, 12), (12090, 12), (13000, 13), (13040, 13)])
    def test_cuda_path(self, monkeypatch, tmp_path, cuda_version, expected):
        self._write_cuda_h(
            tmp_path / 'include',
            f'#ifndef __cuda_cuda_h__\n'
            f'#define __cuda_cuda_h__\n'
            f'#define CUDA_VERSION {cuda_version}\n'
            f'#endif\n')
        monkeypatch.setattr(build, 'get_cuda_path', lambda: str(tmp_path))
        ctx = Context('.', _env={}, _argv=[])
        assert build.detect_cuda_major(ctx) == expected

    def test_include_path_takes_precedence(self, monkeypatch, tmp_path):
        self._write_cuda_h(
            tmp_path / 'cuda' / 'include', '#define CUDA_VERSION 12090\n')
        self._write_cuda_h(
            tmp_path / 'extra', '#  define CUDA_VERSION 13040\n')
        monkeypatch.setattr(
            build, 'get_cuda_path', lambda: str(tmp_path / 'cuda'))
        ctx = Context(
            '.', _env={'CUPY_INCLUDE_PATH': str(tmp_path / 'extra')},
            _argv=[])
        assert build.detect_cuda_major(ctx) == 13

    def test_no_cuda(self, monkeypatch):
        monkeypatch.setattr(build, 'get_cuda_path', lambda: None)
        ctx = Context('.', _env={}, _argv=[])
        assert build.detect_cuda_major(ctx) is None

    def test_cuda_h_without_version(self, monkeypatch, tmp_path):
        self._write_cuda_h(tmp_path / 'include', '/* not a real cuda.h */\n')
        monkeypatch.setattr(build, 'get_cuda_path', lambda: str(tmp_path))
        ctx = Context('.', _env={}, _argv=[])
        assert build.detect_cuda_major(ctx) is None


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        ctx = Context('.', _env={}, _argv=[])
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting(ctx, test_hip)

    @pytest.mark.skipif(not test_hip, reason='For ROCm/HIP environment')
    def test_check_hip_version(self):
        with self.assertRaises(RuntimeError):
            build.get_hip_version()
        assert build.check_hip_version(
            self.compiler, self.settings)
        assert isinstance(build.get_hip_version(), int)
        assert isinstance(build.get_hip_version(True), str)
