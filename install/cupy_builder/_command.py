import os
import sys
from typing import List, Tuple

import setuptools
import setuptools.command.build_ext

import cupy_builder
from cupy_builder._context import Context
from cupy_builder.cupy_setup_build import cythonize
from cupy_builder._compiler import DeviceCompilerUnix, DeviceCompilerWin32


def filter_files_by_extension(
        sources: List[str],
        extension: str,
) -> Tuple[List[str], List[str]]:
    sources_selected = []
    sources_others = []
    for src in sources:
        if os.path.splitext(src)[1] == extension:
            sources_selected.append(src)
        else:
            sources_others.append(src)
    return sources_selected, sources_others


def compile_device_code(
        ctx: Context,
        ext: setuptools.Extension
) -> Tuple[List[str], List[str]]:
    """Compiles device code ("*.cu").

    This method invokes the device compiler (nvcc/hipcc) to build object
    files from device code, then returns the tuple of:
    - list of remaining (non-device) source files ("*.cpp")
    - list of compiled object files for device code ("*.o")
    """
    sources_cu, sources_cpp = filter_files_by_extension(
        ext.sources, '.cu')
    if len(sources_cu) == 0:
        # No device code used in this extension.
        return ext.sources, []

    if sys.platform == 'win32':
        compiler = DeviceCompilerWin32(ctx)
    else:
        compiler = DeviceCompilerUnix(ctx)

    objects = []
    for src in sources_cu:
        print(f'{ext.name}: Device code: {src}')
        obj_ext = 'obj' if sys.platform == 'win32' else 'o'
        # TODO(kmaehashi): embed CUDA version in path
        obj = f'build/temp.device_objects/{src}.{obj_ext}'
        if os.path.exists(obj) and (_get_timestamp(src) < _get_timestamp(obj)):
            print(f'{ext.name}: Reusing cached object file: {obj}')
        else:
            os.makedirs(os.path.dirname(obj), exist_ok=True)
            print(f'{ext.name}: Building: {obj}')
            compiler.compile(obj, src, ext)
        objects.append(obj)

    return sources_cpp, objects


def _get_timestamp(path: str) -> float:
    stat = os.lstat(path)
    return max(stat.st_atime, stat.st_mtime, stat.st_ctime)


class custom_build_ext(setuptools.command.build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def build_extensions(self) -> None:
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs
            if hasattr(self.compiler, 'initialize'):
                # Workarounds a bug in setuptools/distutils on Windows by
                # initializing the compiler before starting a thread.
                # By default, MSVCCompiler performs initialization in the
                # first compilation. However, in parallel compilation mode,
                # the init code runs in each thread and messes up the internal
                # state as the init code is not locked and is not idempotent.
                # https://github.com/pypa/setuptools/blob/v60.0.0/setuptools/_distutils/_msvccompiler.py#L322-L327
                self.compiler.initialize()

        # Compile "*.pyx" files into "*.cpp" files.
        print('Cythonizing...')
        cythonize(self.extensions, cupy_builder.get_context())

        # Change an extension in each source filenames from "*.pyx" to "*.cpp".
        # c.f. `Cython.Distutils.old_build_ext`
        for ext in self.extensions:
            sources_pyx, sources_others = filter_files_by_extension(
                ext.sources, '.pyx')
            sources_cpp = ['{}.cpp'.format(os.path.splitext(src)[0])
                           for src in sources_pyx]
            ext.sources = sources_cpp + sources_others
            for src in ext.sources:
                if not os.path.isfile(src):
                    raise RuntimeError(f'Fatal error: missing file: {src}')

        print('Building extensions...')
        super().build_extensions()

    def build_extension(self, ext: setuptools.Extension) -> None:
        ctx = cupy_builder.get_context()

        # Compile "*.cu" files into object files.
        sources_cpp, extra_objects = compile_device_code(ctx, ext)

        # Remove device code from list of sources, and instead add compiled
        # object files to link.
        ext.sources = sources_cpp
        ext.extra_objects += extra_objects

        # Let setuptools do the rest of the build process, i.e., compile
        # "*.cpp" files and link object files generated from "*.cu".
        super().build_extension(ext)
