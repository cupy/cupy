from __future__ import annotations

from distutils import ccompiler
import os
import subprocess
import sys

import pytest
from setuptools import Extension

from cupy_builder._compiler import DeviceCompilerWin32
from cupy_builder._context import Context
from cupy_builder import install_build as build


@pytest.mark.skipif(
    sys.platform != 'win32',
    reason='Requires a native Windows HIP build and GPU')
def test_windows_hip_compiled_object_executes(tmp_path):
    cupy = pytest.importorskip('cupy')
    if not cupy.cuda.runtime.is_hip:
        pytest.skip('Requires the HIP backend')
    ctx = Context('.', _env={'CUPY_INSTALL_USE_HIP': '1'}, _argv=[])
    toolkit = build.get_rocm_path()
    assert toolkit, 'Set ROCM_HOME or HIP_PATH to the Windows HIP toolkit'
    source = tmp_path / 'device source.cu'
    obj = tmp_path / 'device object.obj'
    executable = tmp_path / 'device executable.exe'
    source.write_text(r'''
        #include <hip/hip_runtime.h>
        #include <stdio.h>
        #if __cplusplus < 201703L
        #error CuPy device objects require C++17
        #endif
        CYTHON_EXTERN_C int host_answer() { return 42; }
        __global__ void answer(int* value) { *value = 42; }
        int main() {
            int* device = nullptr;
            if (hipMalloc(&device, sizeof(int)) != hipSuccess) return 1;
            hipLaunchKernelGGL(answer, dim3(1), dim3(1), 0, 0, device);
            if (hipGetLastError() != hipSuccess) return 2;
            int value = 0;
            if (hipMemcpy(&value, device, sizeof(int),
                          hipMemcpyDeviceToHost) != hipSuccess) return 3;
            if (hipFree(device) != hipSuccess) return 4;
            printf("%d", value);
            return value == host_answer() ? 0 : 5;
        }
    ''', encoding='utf-8')
    ext = Extension('device', [str(source)], include_dirs=[
        os.path.join(toolkit, 'include'),
        os.path.join(toolkit, 'include', 'hip'),
    ], define_macros=[('__HIP_PLATFORM_AMD__', '1'),
                      ('CYTHON_EXTERN_C', 'extern "C"')])
    DeviceCompilerWin32(ctx).compile(str(obj), str(source), ext)
    host = ccompiler.new_compiler()
    host.initialize()
    host.link_executable(
        [str(obj)], str(executable), libraries=['amdhip64'],
        library_dirs=[os.path.join(toolkit, 'lib')], target_lang='c++')
    child_env = dict(os.environ)
    child_env['PATH'] = (os.path.join(toolkit, 'bin') + os.pathsep +
                         child_env.get('PATH', ''))
    result = subprocess.run(
        [str(executable)], env=child_env, capture_output=True, text=True,
        check=True)
    assert result.stdout == '42'


def test_windows_hip_compiler_missing(monkeypatch, tmp_path):
    ctx = Context('.', _env={'CUPY_INSTALL_USE_HIP': '1'}, _argv=[])
    monkeypatch.setattr(build, '_rocm_path', 'NOT_INITIALIZED')
    monkeypatch.delenv('HIPCC', raising=False)
    monkeypatch.delenv('HIP_CLANG_PATH', raising=False)
    monkeypatch.setenv('ROCM_HOME', str(tmp_path))
    monkeypatch.setenv('HIP_PATH', str(tmp_path))
    with pytest.raises(RuntimeError, match='Windows HIP Clang.*not be found'):
        DeviceCompilerWin32(ctx).compile(
            'example.obj', 'example.cu', Extension('example', ['example.cu']))


@pytest.mark.parametrize('layout', [
    ('lib', 'llvm', 'bin'), ('llvm', 'bin'), ('bin',),
])
def test_windows_hip_clang_discovery(monkeypatch, tmp_path, layout):
    toolkit = tmp_path / 'HIP SDK'
    binary = toolkit.joinpath(*layout, 'clang.exe')
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(build, 'PLATFORM_WIN32', True)
    monkeypatch.setattr(build, '_rocm_path', 'NOT_INITIALIZED')
    monkeypatch.delenv('ROCM_HOME', raising=False)
    monkeypatch.delenv('HIP_CLANG_PATH', raising=False)
    monkeypatch.setenv('HIP_PATH', str(toolkit))
    assert build.get_rocm_path() == str(toolkit)
    assert build.get_hip_clang_path() == [str(binary)]


def test_windows_hip_clang_directory_override(monkeypatch, tmp_path):
    binary = tmp_path / 'LLVM tools' / 'clang.exe'
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setenv('HIP_CLANG_PATH', str(binary.parent))
    assert build.get_hip_clang_path() == [str(binary)]
    binary.unlink()
    binary.mkdir()
    assert build.get_hip_clang_path() is None




def test_windows_hip_clang_requires_toolkit_root(monkeypatch, tmp_path):
    binary = tmp_path / 'LLVM tools' / 'clang.exe'
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setenv('HIP_CLANG_PATH', str(binary.parent))
    monkeypatch.setattr(build, '_rocm_path', '')
    ctx = Context('.', _env={'CUPY_INSTALL_USE_HIP': '1'}, _argv=[])
    with pytest.raises(RuntimeError, match='Windows HIP toolkit root'):
        DeviceCompilerWin32(ctx).compile(
            'example.obj', 'example.cu', Extension('example', ['example.cu']))


@pytest.mark.skipif(
    sys.platform != 'win32',
    reason='Requires native Windows HIPRTC and GPU')
def test_windows_hiprtc_spaced_source_path_executes(tmp_path):
    cupy = pytest.importorskip('cupy')
    if not cupy.cuda.runtime.is_hip:
        pytest.skip('Requires the HIP backend')
    from cupy.cuda import compiler, function

    source = 'extern "C" __global__ void answer(int* value) { *value = 42; }'
    binary, _ = compiler.compile_using_nvrtc(
        source, options=('--std=c++17',),
        filename=str(tmp_path / 'virtual source with spaces.cu'))
    module = function.Module()
    module.load(binary)
    result = cupy.zeros(1, dtype=cupy.int32)
    module.get_function('answer')((1,), (1,), (result,))
    cupy.cuda.runtime.deviceSynchronize()
    assert result.get().tolist() == [42]


@pytest.mark.skipif(
    sys.platform != 'win32', reason='Native Windows HIP compiler')
def test_windows_explicit_hipcc_runtime_backend_executes():
    cupy = pytest.importorskip('cupy')
    if not cupy.cuda.runtime.is_hip:
        pytest.skip('Requires the HIP backend')
    module = cupy.RawModule(
        code='extern "C" __global__ void answer(int* value) { *value = 42; }',
        backend='nvcc')
    result = cupy.zeros(1, dtype=cupy.int32)
    module.get_function('answer')((1,), (1,), (result,))
    cupy.cuda.runtime.deviceSynchronize()
    assert result.get().tolist() == [42]


