from distutils import ccompiler
from distutils import errors
import os

from setuptools.command import build_ext

import cupy_builder
from cupy_builder.cupy_setup_build import cythonize
from cupy_builder.cupy_setup_build import check_extensions
from cupy_builder.cupy_setup_build import get_ext_modules
import cupy_builder.install_build as build
from cupy_builder._compiler import _UnixCCompiler, _MSVCCompiler


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if (build.get_nvcc_path() is not None
                or build.get_hipcc_path() is not None):
            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        if not build.PLATFORM_WIN32:
                            CCompiler = _UnixCCompiler
                        else:
                            CCompiler = _MSVCCompiler
                        return CCompiler(
                            None, kwargs['dry_run'], kwargs['force'])
                return _wrap_new_compiler
            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            # Intentionally causes DistutilsPlatformError in
            # ccompiler.new_compiler() function to hook.
            self.compiler = 'nvidia'
        ctx = cupy_builder.get_context()
        ext_modules = get_ext_modules(True, ctx)  # get .pyx modules
        cythonize(ext_modules, ctx)
        check_extensions(self.extensions)
        build_ext.build_ext.run(self)

    def build_extensions(self):
        num_jobs = int(os.environ.get('CUPY_NUM_BUILD_JOBS', '4'))
        if num_jobs > 1:
            self.parallel = num_jobs
        super().build_extensions()
