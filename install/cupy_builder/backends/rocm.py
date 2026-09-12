"""AMD ROCm / HIP backend descriptor.

ROCm reuses the CUDA-shaped sources but has its own compiler (``hipcc``),
a deeply nested include layout and a different C++ standard requirement.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build

from cupy_builder.backends._base import Backend

if TYPE_CHECKING:
    from cupy_builder._context import Context


class RocmBackend(Backend):
    name = 'rocm'
    env_flag = 'CUPY_INSTALL_USE_HIP'
    version_macro = 'CUPY_HIP_VERSION'
    sdk_env_var = 'ROCM_HOME'
    compiler_env_var = 'HIPCC'
    compiler_name = 'hipcc'
    needs_cub_headers = True  # hipCUB

    #: Minimum supported HIP version, encoded as ``major * 100 + minor``.
    minimum_version = 305

    def get_sdk_path(self) -> str | None:
        return build.get_rocm_path()

    def get_device_compiler(self) -> list[str] | None:
        return build.get_hipcc_path()

    def get_include_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        rels = [
            'include',
            'include/hip',
            'include/rocrand',
            'include/hiprand',
            'include/roctracer',
            'include/hipblas',
            'include/hipsparse',
            'include/hipfft',
            'include/rocsolver',
            'include/rccl',
        ]
        return [os.path.join(sdk, r) for r in rels]

    def get_library_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        return [os.path.join(sdk, 'lib')]

    def get_extra_compile_args(self) -> list[str]:
        # ROCm 5.3 and above requires C++14
        return ['-std=c++14']

    def get_define_macros(self) -> list[tuple[str, str]]:
        return [
            ('CUPY_USE_HIP', '1'),
            # introduced since ROCm 4.2.0
            ('__HIP_PLATFORM_AMD__', '1'),
            # deprecated since ROCm 4.2.0
            ('__HIP_PLATFORM_HCC__', '1'),
            # Fix for ROCm 6.3.0, see ROCm/rocThrust#502
            ('THRUST_DEVICE_SYSTEM', 'THRUST_DEVICE_SYSTEM_HIP'),
        ]

    def get_device_compile_args(self, ctx: Context, src: str) -> list[str]:
        compiler = self.get_device_compiler()
        if compiler is None:
            raise RuntimeError('hipcc not found under ROCm path: %s'
                               % self.get_sdk_path())
        base_opts = build.get_compiler_base_options(compiler)
        return compiler + base_opts + [
            '-O2', '-fPIC', '--include', 'hip_runtime.h', '--std=c++17']

    def get_compile_time_env(self, ctx: Context) -> dict[str, Any]:
        return {
            'CUPY_CUDA_VERSION': 0,
            'CUPY_CANN_VERSION': 0,
            'CUPY_HIP_VERSION': self.get_version(),
        }

    def supports_platform(self, platform: str) -> bool:
        # ROCm is supported on Linux only.
        return platform == 'linux'

    def check_version(self, compiler: Any, settings: Any) -> bool:
        return build.check_hip_version(compiler, settings)

    def get_version(self) -> int:
        return build.get_hip_version()
