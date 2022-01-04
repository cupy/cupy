import os
import sys

import setuptools
import setuptools.command.build_ext

import cupy_builder
from cupy_builder._context import Context
from cupy_builder.cupy_setup_build import cythonize
from cupy_builder.cupy_setup_build import check_extensions
from cupy_builder.cupy_setup_build import get_ext_modules
from cupy_builder._compiler import DeviceCompilerUnix, DeviceCompilerWin32


def compile_device_code(ctx: Context, ext: setuptools.Extension):
    """Compiles device code ("*.cu").

    This method invokes the device compiler (nvcc/hipcc) to build an object
    file from the device code, then modify the given extension to link the
    object files in later step.
    """
    sources_cu = []
    sources_cpp = []
    for src in ext.sources:
        if os.path.splitext(src)[1] == '.cu':
            sources_cu.append(src)
        else:
            sources_cpp.append(src)
    if len(sources_cu) == 0:
        # No device code used in this extension.
        return ext.sources, []

    if sys.platform == 'win32':
        compiler = DeviceCompilerWin32(ctx)
    else:
        compiler = DeviceCompilerUnix(ctx)

    objects = []
    for src in sources_cu:
        print('Compiling device code', src, 'for module', ext.name)
        obj = os.path.abspath(f'build/temp.device_objects/{src}.o')
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        # TODO: check timestamp
        compiler.compile(obj, src, ext)
        objects.append(obj)

    return sources_cpp, objects


class custom_build_ext(setuptools.command.build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        ctx = cupy_builder.get_context()
        ext_modules = get_ext_modules(True, ctx)  # get .pyx modules
        cythonize(ext_modules, ctx)
        check_extensions(self.extensions)
        super().run()

    def build_extensions(self):
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs
        super().build_extensions()

    def build_extension(self, ext):
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
