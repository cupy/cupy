"""NVIDIA CUDA backend descriptor.

Owns the CUDA Toolkit layout, the ``nvcc`` code-generation flags
(``_nvcc_gencode_options``) and the CUDA version probe.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build

from cupy_builder.backends._base import Backend

if TYPE_CHECKING:
    from cupy_builder._context import Context


def nvcc_gencode_options(cuda_version: int) -> list[str]:
    """Return NVCC ``--generate-code`` options for ``cuda_version``."""
    if sys.argv == ['setup.py', 'develop']:
        return []

    envcfg = os.getenv('CUPY_NVCC_GENERATE_CODE', None)
    if envcfg is not None and envcfg != 'current':
        return ['--generate-code={}'.format(arch)
                for arch in envcfg.split(';') if len(arch) > 0]
    if envcfg == 'current' and build.get_compute_capabilities() is not None:
        ccs = build.get_compute_capabilities()
        arch_list = [
            f'compute_{cc}' if cc < 60 else (f'compute_{cc}', f'sm_{cc}')
            for cc in ccs]
    else:
        # The arch_list specifies virtual architectures, such as 'compute_61',
        # and real architectures, such as 'sm_61', for which the CUDA
        # input files are to be compiled.
        #
        # See the CUDA docs for the list of supported architectures:
        #   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
        #
        # CuPy utilizes CUDA Minor Version Compatibility to support all CUDA
        # minor versions in a single binary package (e.g., `cupy-cuda12x`).
        # CUBIN must be generated for all supported compute capabilities; PTX
        # for the latest architecture is also included as a fallback.
        aarch64 = (platform.machine() == 'aarch64')
        if cuda_version >= 13000:
            arch_list = [('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),
                         ('compute_100f', 'sm_100'),
                         ('compute_120f', 'sm_120'),
                         'compute_120']
            if aarch64:
                arch_list += [
                    ('compute_87', 'sm_87'),    # Jetson (Orin)
                    ('compute_110', 'sm_110'),  # Jetson (Thor)
                ]
        elif cuda_version >= 12000:
            arch_list = [('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),]
            if cuda_version < 12080:
                arch_list.append('compute_90')
            elif 12080 <= cuda_version < 12090:
                arch_list += [('compute_100', 'sm_100'),
                              ('compute_120', 'sm_120'),
                              'compute_100']
            elif 12090 <= cuda_version:
                arch_list += [('compute_100f', 'sm_100'),
                              ('compute_120f', 'sm_120'),
                              'compute_100']

            if aarch64:
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11080:
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         ('compute_89', 'sm_89'),
                         ('compute_90', 'sm_90'),
                         'compute_90']
            if aarch64:
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11040:
            arch_list = [('compute_35', 'sm_35'),
                         ('compute_37', 'sm_37'),
                         ('compute_50', 'sm_50'),
                         ('compute_52', 'sm_52'),
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
            if aarch64:
                arch_list += [
                    ('compute_72', 'sm_72'),  # Jetson (Xavier)
                    ('compute_87', 'sm_87'),  # Jetson (Orin)
                ]
        elif cuda_version >= 11020:
            arch_list = ['compute_35',
                         'compute_50',
                         ('compute_60', 'sm_60'),
                         ('compute_61', 'sm_61'),
                         ('compute_70', 'sm_70'),
                         ('compute_75', 'sm_75'),
                         ('compute_80', 'sm_80'),
                         ('compute_86', 'sm_86'),
                         'compute_86']
        else:
            # This should not happen.
            assert False

    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            virtual_arch, real_arch = arch
            options.append('--generate-code=arch={},code={}'.format(
                virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(
                arch, arch))

    return options


class CudaBackend(Backend):
    name = 'cuda'
    env_flag = ''                 # CUDA is the default backend
    version_macro = 'CUPY_CUDA_VERSION'
    sdk_env_var = 'CUDA_PATH'
    compiler_env_var = 'NVCC'
    compiler_name = 'nvcc'
    needs_cub_headers = True

    def get_sdk_path(self) -> str | None:
        return build.get_cuda_path()

    def get_device_compiler(self) -> list[str] | None:
        return build.get_nvcc_path()

    def get_include_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        return [os.path.join(sdk, 'include')]

    def get_library_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        if build.PLATFORM_WIN32:
            return [os.path.join(sdk, 'bin'), os.path.join(sdk, 'lib', 'x64')]
        return [os.path.join(sdk, 'lib64'), os.path.join(sdk, 'lib')]

    def get_device_compile_args(self, ctx: Context, src: str) -> list[str]:
        compiler = self.get_device_compiler()
        if compiler is None:
            raise RuntimeError('nvcc not found under CUDA path: %s'
                               % self.get_sdk_path())
        base_opts = build.get_compiler_base_options(compiler)
        cuda_version = ctx.features['cuda'].get_version()
        postargs = nvcc_gencode_options(cuda_version) + [
            '-Xfatbin=-compress-all', '-O2', '--compiler-options="-fPIC"',
            '--expt-relaxed-constexpr']
        num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
        postargs += ['--std=c++17',
                     f'-t{num_threads}',
                     '-Xcompiler=-fno-gnu-unique']
        return compiler + base_opts + postargs

    def get_compile_time_env(self, ctx: Context) -> dict[str, Any]:
        return {
            'CUPY_CUDA_VERSION': ctx.features['cuda'].get_version(),
            'CUPY_HIP_VERSION': 0,
            'CUPY_CANN_VERSION': 0,
        }

    def supports_platform(self, platform: str) -> bool:
        return platform in ('linux', 'win32')

    def check_version(self, compiler: Any, settings: Any) -> bool:
        # The CUDA version is detected by the ``CUDA_cuda`` feature's
        # ``configure()``, which compiles and runs a probe via the NVCC
        # toolchain; there is nothing extra to do here.
        return True

    def get_version(self) -> int:
        return self._version_cache
