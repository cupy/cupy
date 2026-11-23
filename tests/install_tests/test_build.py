from __future__ import annotations

from distutils import ccompiler
from distutils import sysconfig
import os
import pathlib
import re
import sys
import tomllib
import types
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

    def test_cuda_version(self, monkeypatch, tmp_path):
        self._write_cuda_h(
            tmp_path / 'include', '#define CUDA_VERSION 13040\n')
        monkeypatch.setattr(build, 'get_cuda_path', lambda: str(tmp_path))
        ctx = Context('.', _env={}, _argv=[])
        assert build.detect_cuda_version(ctx) == 13040

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


class TestGetCudaPythonVersions:

    with open(pathlib.Path(__file__).parents[2] / 'pyproject.toml', 'rb') as f:
        floors = tomllib.load(f)['tool']['cupy']['cuda-python']
    bindings_floor_12 = floors['cuda-bindings']['12']
    nvmath_floor = floors['nvmath-python']

    @staticmethod
    def _install(monkeypatch, cuda_bindings_version,
                 nvmath_version=nvmath_floor, nvmath=True):
        cuda_bindings = types.ModuleType('cuda.bindings')
        cuda_bindings.__version__ = cuda_bindings_version
        monkeypatch.setitem(sys.modules, 'cuda.bindings', cuda_bindings)
        nvmath_module = types.ModuleType('nvmath')
        nvmath_module.__version__ = nvmath_version
        monkeypatch.setitem(sys.modules, 'nvmath', nvmath_module)
        monkeypatch.setitem(
            sys.modules, 'nvmath.bindings',
            types.ModuleType('nvmath.bindings') if nvmath else None)

    @pytest.mark.parametrize('cuda_major', [12, 13])
    def test_ok(self, monkeypatch, cuda_major):
        bindings_floor = self.floors['cuda-bindings'][str(cuda_major)]
        self._install(monkeypatch, bindings_floor)
        assert build.get_cuda_python_versions(cuda_major, self.floors) == (
            self.nvmath_floor, bindings_floor)

    def test_local_version_dropped(self, monkeypatch):
        self._install(monkeypatch, f'{self.bindings_floor_12}+g1234abc',
                      nvmath_version='1.1.0.dev3+gabcdef0')
        assert build.get_cuda_python_versions(12, self.floors) == (
            '1.1.0.dev3', self.bindings_floor_12)

    @pytest.mark.parametrize('cuda_bindings_version', ['12.9.4', '13.4.4'])
    def test_cuda_bindings_unsupported(
            self, monkeypatch, cuda_bindings_version):
        self._install(monkeypatch, cuda_bindings_version)
        with pytest.raises(RuntimeError, match=re.escape(
                f'cuda-bindings>={self.bindings_floor_12}')):
            build.get_cuda_python_versions(12, self.floors)

    @pytest.mark.parametrize('nvmath_version', ['0.9.0', '2.0.0'])
    def test_nvmath_unsupported(self, monkeypatch, nvmath_version):
        self._install(monkeypatch, self.bindings_floor_12,
                      nvmath_version=nvmath_version)
        with pytest.raises(RuntimeError, match=re.escape(
                f'nvmath-python>={self.nvmath_floor},<2')):
            build.get_cuda_python_versions(12, self.floors)

    def test_missing_nvmath(self, monkeypatch):
        self._install(monkeypatch, self.bindings_floor_12, nvmath=False)
        with pytest.raises(RuntimeError, match='requires nvmath-python'):
            build.get_cuda_python_versions(12, self.floors)


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
