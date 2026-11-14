from __future__ import annotations

from distutils import ccompiler, sysconfig
from distutils.util import get_platform
import os
import shutil

from cupy_builder._context import Context
import cupy_builder.install_build as build


def configure_win32_hip_host_compiler(
        compiler, ctx: Context, plat_name: str | None = None) -> None:
    """Use Clang's MSVC driver with the Python toolchain's CRT and linker."""
    if not (build.PLATFORM_WIN32 and ctx.use_hip):
        return
    if compiler.compiler_type != 'msvc':
        raise RuntimeError('Windows HIP requires the MSVC-compatible host '
                           'compiler toolchain (do not select MinGW).')

    override = os.environ.get('CUPY_HIP_HOST_COMPILER')
    if override:
        executable = override.strip('"')
        executable = (executable if os.path.isfile(executable) else
                      shutil.which(executable))
    else:
        command = build.get_hip_clang_path()
        executable = command[0] if command else None
    if not executable:
        raise RuntimeError(
            'Clang could not be found for the Windows HIP host compiler. '
            'Install LLVM in the HIP toolkit or set CUPY_HIP_HOST_COMPILER '
            'to the full path of clang-cl.exe or clang.exe (no arguments).')

    target = {
        'win32': 'i686-pc-windows-msvc',
        'win-amd64': 'x86_64-pc-windows-msvc',
        'win-arm64': 'aarch64-pc-windows-msvc',
    }.get(plat_name or get_platform())
    if target is None:
        raise RuntimeError('Unsupported Windows HIP host target: '
                           f'{plat_name or get_platform()}')

    # Keep distutils' Visual Studio/Windows SDK discovery, include/lib paths,
    # link.exe, lib.exe and manifest handling. Only the C/C++ frontend changes.
    if not compiler.initialized:
        compiler.initialize(plat_name)
    ctx.win32_cl_exe_path = compiler.cc
    compiler.cc = executable
    for options in (compiler.compile_options, compiler.compile_options_debug):
        # MSVC's /GL bitcode cannot be mixed with LLVM or device COFF objects.
        options[:] = [option for option in options if option != '/GL']
        options[:0] = ['--driver-mode=cl', f'--target={target}']
    for name in (
            'ldflags_exe', 'ldflags_exe_debug',
            'ldflags_shared', 'ldflags_shared_debug',
            'ldflags_static', 'ldflags_static_debug'):
        options = getattr(compiler, name)
        options[:] = [option for option in options if option != '/LTCG']
    print(f'Windows HIP host compiler: {compiler.cc} (target {target})')
    print(f'Windows HIP host linker: {compiler.linker}')


def new_host_compiler(ctx: Context):
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    configure_win32_hip_host_compiler(compiler, ctx)
    return compiler
