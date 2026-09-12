"""Huawei Ascend / CANN backend descriptor.

CANN lays its SDK out differently from CUDA/ROCm and renames libraries and
header directories between releases; all of that knowledge lives here so the
rest of the build system stays backend-agnostic.

Key SDK layout (CANN 8.5.x)::

    <CANN>/
        include/                 # acl/acl.h, aclnn/...
        include/aclnn/
        x86_64-linux/pkg_inc/    # base/dlog_pub.h (needed since 8.5)
        lib64/                   # libascendcl.so, libopapi*.so, ...
        runtime/lib64/
        compiler/ccec_compiler/bin/bisheng   # device compiler

The NNAL toolkit (BLAS/FFT/asdsip) is an optional, separately installed
package; when present its include/lib dirs are appended.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build

from cupy_builder.backends._base import Backend

if TYPE_CHECKING:
    from cupy_builder._context import Context


class AscendBackend(Backend):
    name = 'ascend'
    env_flag = 'CUPY_INSTALL_USE_ASCEND'
    version_macro = 'CUPY_CANN_VERSION'
    sdk_env_var = 'ASCEND_HOME_PATH'
    compiler_env_var = ''          # CANN has no standard override variable
    compiler_name = 'bisheng (ascendcc)'

    #: Minimum supported CANN version, encoded as major*100 + minor*10 + patch.
    minimum_version = 820

    #: CANN is a user-installed, relocatable toolkit (multi-GB, installed
    #: *after* the wheel). Its absolute path must never be baked into a
    #: redistributable extension module, or the wheel only ever imports on
    #: the machine that built it.
    embed_sdk_in_rpath = False

    def get_sdk_path(self) -> str | None:
        return build.get_cann_path()

    def get_device_compiler(self) -> list[str] | None:
        return build.get_ascendcc_path()

    def get_include_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        dirs = [
            os.path.join(sdk, 'include'),
            os.path.join(sdk, 'include/aclnn'),
            # CANN 8.5 needs `base/dlog_pub.h` from pkg_inc
            os.path.join(sdk, 'x86_64-linux/pkg_inc'),
            os.path.join(sdk, 'include/experiment/platform'),
        ]
        dirs += self._nnal_dirs(sdk, 'include')
        return [d for d in dirs if os.path.isdir(d)]

    def get_library_dirs(self, ctx: Context) -> list[str]:
        sdk = self.get_sdk_path()
        if not sdk:
            return []
        dirs = [
            os.path.join(sdk, 'lib64'),
            os.path.join(sdk, 'runtime/lib64'),
        ]
        dirs += self._nnal_dirs(sdk, 'lib')
        return [d for d in dirs if os.path.isdir(d)]

    def get_extra_compile_args(self) -> list[str]:
        return ['-std=c++17']

    def get_define_macros(self) -> list[tuple[str, str]]:
        return [
            ('CUPY_USE_ASCEND', '1'),
            # keep in sync with the Cython compile-time constant
            ('CUPY_CANN_VERSION', str(self.get_version())),
        ]

    def get_device_compile_args(self, ctx: Context, src: str) -> list[str]:
        compiler = self.get_device_compiler()
        if compiler is None:
            raise RuntimeError(
                'Ascend device compiler (bisheng) not found under CANN path: %s'
                % self.get_sdk_path())
        base_opts = build.get_compiler_base_options(compiler)
        return compiler + base_opts + [
            '-O2', '-fPIC', '--include', 'kernel_operator.h', '--std=c++17']

    def get_compile_time_env(self, ctx: Context) -> dict[str, Any]:
        return {
            'CUPY_CUDA_VERSION': 0,
            'CUPY_HIP_VERSION': 0,
            'CUPY_CANN_VERSION': self.get_version(),
        }

    def supports_platform(self, platform: str) -> bool:
        # The can be lifted once the CANN Windows toolchain is supported.
        return platform == 'linux'

    def check_version(self, compiler: Any, settings: Any) -> bool:
        return build.check_cann_version(compiler, settings)

    def get_version(self) -> int:
        return build.get_cann_version()

    # ------------------------------------------------------------------
    # Wheel identity
    # ------------------------------------------------------------------
    def get_wheel_platform_tag(self) -> str | None:
        """Return e.g. ``'cann8.5'`` so wheels for different CANN majors/minors
        can not be confused with one another.

        ``aclnn`` operator signatures change between CANN releases and
        ``libop_common.so``/``liboptiling.so`` are coupled per release, so a
        binary built against 8.5 is *not* a drop-in replacement for 9.0.
        """
        version = self.get_version()
        if version in (self.NOT_AVAILABLE, 0):
            return None
        major, rest = divmod(version, 100)
        minor = rest // 10
        return f'cann{major}.{minor}'

    def get_wheel_metadata(self) -> dict[str, Any]:
        """Record the exact CANN version the wheel was built against.

        ``cupy/.data/_wheel.json`` is written into the wheel and re-checked at
        import time, so a version mismatch produces an actionable error rather
        than a segfault or an ``undefined symbol`` traceback.
        """
        version = self.get_version()
        sdk = self.get_sdk_path()
        metadata: dict[str, Any] = {
            'cupy_backend': self.name,
            # raw encoded version, e.g. 851 for CANN 8.5.1
            'cann_version': version,
            'cann_version_str': build.format_cann_version(version),
        }
        if sdk:
            metadata['cann_build_path'] = sdk
        return metadata

    # ------------------------------------------------------------------
    @staticmethod
    def _nnal_dirs(sdk: str, leaf: str) -> list[str]:
        """NNAL is a sibling install (``<CANN>/../../nnal``) when present."""
        return [
            os.path.join(sdk, '../../nnal/asdsip/latest', leaf),
        ]
