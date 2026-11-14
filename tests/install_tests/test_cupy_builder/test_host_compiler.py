from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cupy_builder._context import Context
from cupy_builder import _host_compiler
from cupy_builder import install_build as build


def _msvc_compiler():
    compiler = SimpleNamespace(
        compiler_type='msvc', initialized=False,
        cc='cl.exe', linker='link.exe', lib='lib.exe',
        compile_options=['/nologo', '/O2', '/GL', '/MD'],
        compile_options_debug=['/nologo', '/Od', '/MDd'],
        initialize_calls=[],
    )
    for name in (
            'ldflags_exe', 'ldflags_exe_debug',
            'ldflags_shared', 'ldflags_shared_debug',
            'ldflags_static', 'ldflags_static_debug'):
        setattr(compiler, name, ['/nologo', '/LTCG'])
    compiler._ldflags = {'shared': compiler.ldflags_shared}

    def initialize(plat_name=None):
        compiler.initialize_calls.append(plat_name)
        compiler.initialized = True

    compiler.initialize = initialize
    return compiler




@pytest.mark.parametrize('override', [None, 'missing-clang.exe'])
def test_windows_hip_missing_clang(monkeypatch, override):
    monkeypatch.setattr(build, 'PLATFORM_WIN32', True)
    monkeypatch.setattr(build, 'get_hip_clang_path', lambda: None)
    monkeypatch.setattr(_host_compiler.shutil, 'which', lambda path: None)
    if override is None:
        monkeypatch.delenv('CUPY_HIP_HOST_COMPILER', raising=False)
    else:
        monkeypatch.setenv('CUPY_HIP_HOST_COMPILER', override)
    compiler = _msvc_compiler()
    with pytest.raises(RuntimeError, match='CUPY_HIP_HOST_COMPILER'):
        _host_compiler.configure_win32_hip_host_compiler(
            compiler, SimpleNamespace(use_hip=True), 'win-amd64')
    assert compiler.initialize_calls == []
    assert compiler.cc == 'cl.exe'




def test_windows_hip_rejects_mingw(monkeypatch):
    monkeypatch.setattr(build, 'PLATFORM_WIN32', True)
    with pytest.raises(RuntimeError, match='do not select MinGW'):
        _host_compiler.configure_win32_hip_host_compiler(
            SimpleNamespace(compiler_type='mingw32'),
            SimpleNamespace(use_hip=True))




@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows HIP host')
def test_windows_hip_host_headers_and_vectors(monkeypatch):
    import cupy_builder
    import sysconfig

    if not build.get_rocm_path():
        pytest.skip('Requires a Windows HIP toolkit')
    ctx = Context('.', _env={'CUPY_INSTALL_USE_HIP': '1'}, _argv=[])
    monkeypatch.setattr(cupy_builder, 'get_context', lambda: ctx)
    compiler = _host_compiler.new_host_compiler(ctx)
    settings = build.get_compiler_setting(ctx, True)
    settings['include_dirs'].append(sysconfig.get_path('include'))
    output = build.build_and_run(compiler, r'''
        #include <Python.h>
        #include <hip/hip_runtime.h>
        #include <stdio.h>
        #ifndef __clang__
        #error Windows HIP host extensions require Clang
        #endif
        #ifndef _MSC_VER
        #error Windows HIP host extensions require the MSVC ABI
        #endif
        #if !defined(_MT) || !defined(_DLL)
        #error Windows HIP host extensions require the dynamic MSVC CRT
        #endif
        #if __cplusplus < 201703L
        #error Windows HIP host extensions require C++17
        #endif
        int main() {
            int2 value = make_int2(20, 22);
            printf("%d", value.x + value.y);
            return 0;
        }
    ''', include_dirs=settings['include_dirs'],
        define_macros=[('__HIP_PLATFORM_AMD__', '1')],
        extra_compile_args=settings['extra_compile_args'],
        library_dirs=[str(Path(sys.base_prefix) / 'libs')])
    assert output == b'42'
