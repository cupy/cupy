from __future__ import annotations

import sys
import types

import pytest

from cupy_builder._command import check_cuda_python, filter_files_by_extension


def test_filter_files_by_extension():
    sources_cpp = ['a.cpp', 'b.cpp']
    sources_pyx = ['c.pyx']
    sources = sources_cpp + sources_pyx
    assert filter_files_by_extension(
        sources, '.cpp') == (sources_cpp, sources_pyx)
    assert filter_files_by_extension(
        sources, '.pyx') == (sources_pyx, sources_cpp)
    assert filter_files_by_extension(
        sources, '.cu') == ([], sources)
    assert filter_files_by_extension(
        sources_cpp, '.cpp') == (sources_cpp, [])
    assert filter_files_by_extension(
        sources_cpp, '.pyx') == ([], sources_cpp)


class TestCheckCudaPython:

    @staticmethod
    def _install(monkeypatch, cuda_bindings_version, nvmath=True):
        cuda_bindings = types.ModuleType('cuda.bindings')
        cuda_bindings.__version__ = cuda_bindings_version
        monkeypatch.setitem(sys.modules, 'cuda.bindings', cuda_bindings)
        monkeypatch.setitem(
            sys.modules, 'nvmath.bindings',
            types.ModuleType('nvmath.bindings') if nvmath else None)

    @pytest.mark.parametrize(('cuda_version', 'cuda_bindings_version'), [
        (12000, '12.9.4'), (12090, '12.8.0'), (13000, '13.4.4')])
    def test_ok(self, monkeypatch, cuda_version, cuda_bindings_version):
        self._install(monkeypatch, cuda_bindings_version)
        check_cuda_python(cuda_version)

    def test_major_mismatch(self, monkeypatch):
        self._install(monkeypatch, '13.4.4')
        with pytest.raises(RuntimeError, match='does not match CUDA 12.x'):
            check_cuda_python(12090)

    def test_missing_nvmath(self, monkeypatch):
        self._install(monkeypatch, '12.9.4', nvmath=False)
        with pytest.raises(RuntimeError, match='requires nvmath-python'):
            check_cuda_python(12090)

    def test_missing_cuda_bindings(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'cuda.bindings', None)
        with pytest.raises(RuntimeError, match=r'cuda-bindings==12\.\*'):
            check_cuda_python(12090)
