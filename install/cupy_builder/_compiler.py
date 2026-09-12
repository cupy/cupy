from __future__ import annotations

import distutils.ccompiler
import os
import os.path
import platform
import shutil
import subprocess
from typing import Any

from setuptools import Extension

from cupy_builder._context import Context
from cupy_builder.backends.cuda import nvcc_gencode_options


class DeviceCompilerBase:
    """A class that invokes NVCC or HIPCC."""
    _context: Context

    def __init__(self, ctx: Context) -> None:
        self._context = ctx

    def _get_preprocess_options(self, ext: Extension) -> list[str]:
        # https://setuptools.pypa.io/en/latest/deprecated/distutils/apiref.html#distutils.core.Extension
        # https://github.com/pypa/setuptools/blob/v60.0.0/setuptools/_distutils/command/build_ext.py#L524-L526
        incdirs = ext.include_dirs[:]
        macros: list[Any] = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        return distutils.ccompiler.gen_preprocess_options(macros, incdirs)

    def spawn(self, commands: list[str]) -> None:
        print('Command:', commands)
        subprocess.check_call(commands)


class DeviceCompilerUnix(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        # All backend-specific compiler flags live in the backend descriptor,
        # so this dispatches uniformly instead of knowing each backend.
        from cupy_builder.backends import get_backend

        backend = get_backend(self._context)
        cc_args = self._get_preprocess_options(ext) + ['-c']
        device_args = backend.get_device_compile_args(self._context, src)
        print('%s options:' % backend.compiler_name, device_args)
        self.spawn(device_args + cc_args + [src, '-o', obj])


class DeviceCompilerWin32(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        from cupy_builder.backends import get_backend

        backend = get_backend(self._context)
        if not backend.supports_platform('win32'):
            raise RuntimeError(
                '%s is not supported on Windows' % backend.name)

        compiler = backend.get_device_compiler()
        cc_args = self._get_preprocess_options(ext) + ['-c']
        cuda_version = self._context.features['cuda'].get_version()
        postargs = nvcc_gencode_options(cuda_version) + [
            '-Xfatbin=-compress-all', '-O2']
        # Note: we only support CUDA 11.2+ since CuPy v13.0.0.
        # MSVC 14.0 (2015) is deprecated for CUDA 11.2 but we need it
        # to build CuPy because some Python versions were built using it.
        # REF: https://wiki.python.org/moin/WindowsCompilers
        postargs += ['-allow-unsupported-compiler']
        # "/bigobj" to silence `fatal error C1128: number of sections exceeded
        # object file format limit`
        postargs += ['-Xcompiler', '/MD /bigobj', '-D_USE_MATH_DEFINES']
        # Bumping C++ standard from C++14 to C++17 for "if constexpr"
        num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
        postargs += ['--std=c++17',
                     f'-t{num_threads}']
        cl_exe_path = self._find_host_compiler_path()
        if cl_exe_path is not None:
            print(f'Using host compiler at {cl_exe_path}')
            postargs += ['--compiler-bindir', cl_exe_path]
        print('NVCC options:', postargs)
        self.spawn(compiler + cc_args + [src, '-o', obj] + postargs)

    def _find_host_compiler_path(self) -> str | None:
        # c.f. cupy.cuda.compiler._get_extra_path_for_msvc
        cl_exe = shutil.which('cl.exe')
        if cl_exe:
            # The compiler is already on PATH, no extra path needed.
            return None

        if self._context.win32_cl_exe_path is not None:
            return self._context.win32_cl_exe_path

        try:
            # See #8568, #8574, #8583.
            import setuptools.msvc
        except Exception:
            print('Warning: cl.exe could not be auto-detected; '
                  'setuptools.msvc could not be imported')
            return None

        vctools: list[str] = setuptools.msvc.EnvironmentInfo(
            platform.machine()).VCTools
        for path in vctools:
            cl_exe = os.path.join(path, 'cl.exe')
            if os.path.exists(cl_exe):
                return path
        print(f'Warning: cl.exe could not be found in {vctools}')
        return None
