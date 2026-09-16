"""Huawei Ascend / CANN backend descriptor.

CANN lays its SDK out differently from CUDA/ROCm and renames libraries and
header directories between releases; all of that knowledge lives here so the
rest of the build system stays backend-agnostic.

Key SDK layout (CANN 8.5.x)::

    <CANN>/
        include/                 # acl/acl.h, aclnn/...
        include/aclnn/
        <arch>-linux/pkg_inc/    # base/dlog_pub.h (needed since 8.5); CANN 9.x
        pkg_inc/                 #   also installs this flat variant
        lib64/                   # libascendcl.so, libopapi*.so, ...
        runtime/lib64/
        compiler/ccec_compiler/bin/bisheng   # device compiler

The NNAL toolkit (BLAS/FFT/asdsip) is an optional, separately installed
package; when present its include/lib dirs are appended.
"""

from __future__ import annotations

import glob
import os
import platform
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build

from cupy_builder.backends._base import Backend

if TYPE_CHECKING:
    from cupy_builder._context import Context


# ---------------------------------------------------------------------------
# CANN SDK layout helpers (shared with features/ascend.py -- keep the NNAL
# detection logic in this one place; features/ascend.py imports from here).
# ---------------------------------------------------------------------------
def cann_arch_name() -> str:
    """CANN's per-architecture directory stem for this host.

    ``platform.machine()`` reports ``x86_64`` / ``aarch64`` (or ``arm64`` in
    some environments) while CANN names its directories ``x86_64-linux`` and
    ``aarch64-linux``.  Cf. ``build.conda_get_target_name`` for a similar
    mapping; kept separate so the CUDA target-name logic stays untouched.
    """
    machine = platform.machine().lower()
    return 'aarch64' if machine in ('aarch64', 'arm64') else 'x86_64'


def cann_arch_dir(sdk: str, *subpath: str) -> str:
    """``<sdk>/<arch>-linux/<subpath>`` for the host architecture."""
    return os.path.join(sdk, cann_arch_name() + '-linux', *subpath)


def cann_pkg_inc_dirs(sdk: str) -> list[str]:
    """Candidate ``pkg_inc`` directories in preference order.

    CANN 8.5 needs ``base/dlog_pub.h`` from ``pkg_inc``.  8.5.x only ships
    the per-architecture layout (``<sdk>/<arch>-linux/pkg_inc``); 9.x also
    installs a flat ``<sdk>/pkg_inc`` (9.0.1 has both).  The per-arch
    directory wins when both exist; a ``*-linux`` glob is the last-resort
    fallback for unexpected architecture directory names.
    """
    dirs = [
        cann_arch_dir(sdk, 'pkg_inc'),
        os.path.join(sdk, 'pkg_inc'),
    ]
    dirs += sorted(glob.glob(os.path.join(sdk, '*-linux', 'pkg_inc')))
    # De-duplicate, preserving order (the glob may re-propose the per-arch
    # directory listed above).
    return list(dict.fromkeys(dirs))


def nnal_root_dirs(cann_path: str) -> list[str]:
    """Candidate NNAL installation roots for a CANN installation.

    NNAL is installed under ``<CANN>/nnal`` (toolkit layout) or as a sibling
    of the CANN directory.
    """
    return [
        os.path.join(cann_path, 'nnal'),
        os.path.join(os.path.dirname(cann_path), 'nnal'),
    ]


def nnal_dirs(cann_path: str, leaf: str) -> list[str]:
    """``lib``/``lib64``/``include`` dirs of a *verified* NNAL install.

    An NNAL directory that exists but is empty (or has no ``lib``/``lib64``
    subtree holding ``*asdsip*`` files) is not an installation; reporting it
    would add include/library dirs that do not exist and could make
    ``_has_nnal`` lie.  ``leaf`` is ``'include'`` or ``'lib'``.
    """
    lib_patterns = ('*asdsip*', '*adsip*')  # libasdsip*.so (+ user spelling)
    dirs: list[str] = []
    for root in nnal_root_dirs(cann_path):
        if not os.path.isdir(root):
            continue
        # lib/lib64 may sit directly under the root or nested
        # (e.g. <root>/asdsip/latest/lib64), so search recursively.
        libdirs = (glob.glob(os.path.join(root, '**', 'lib64'), recursive=True)
                   + glob.glob(os.path.join(root, '**', 'lib'), recursive=True))
        for libdir in libdirs:
            files: list[str] = []
            for pattern in lib_patterns:
                files += glob.glob(os.path.join(libdir, pattern))
            if not any(os.path.isfile(f) for f in files):
                continue
            if leaf in ('lib', 'lib64'):
                dirs.append(libdir)
            else:  # 'include' etc.: the sibling directory of the lib dir
                dirs.append(os.path.join(os.path.dirname(libdir), leaf))
    return list(dict.fromkeys(dirs))


def has_nnal(cann_path: str | None) -> bool:
    """True only when a *usable* NNAL install (asdsip libs) is present."""
    if not cann_path or cann_path == 'NOT_INITIALIZED':
        return False
    return bool(nnal_dirs(cann_path, 'lib'))


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
            # CANN 8.5 needs `base/dlog_pub.h` from pkg_inc; the directory is
            # arch-specific (<arch>-linux/pkg_inc) with a flat 9.x variant.
            *cann_pkg_inc_dirs(sdk),
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
        # optional ops-fft (CANN FFT) library directory, if present
        ops_fft_dir = self._ops_fft_lib_dir()
        if ops_fft_dir and ops_fft_dir not in dirs:
            dirs.append(ops_fft_dir)
        return [d for d in dirs if os.path.isdir(d)]

    @staticmethod
    def _ops_fft_lib_dir() -> str | None:
        """Directory of ``libcann_ops_fft.so`` (optional ops-fft install)."""
        # Imported lazily: features/ascend_fft.py has no backends dependency,
        # but features/ is loaded lazily overall to keep import graphs simple.
        try:
            from cupy_builder.features.ascend_fft import find_ops_fft_lib
        except ImportError:
            return None
        return find_ops_fft_lib()

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
        """NNAL include/lib dirs of a *verified* install (see ``nnal_dirs``).

        NNAL is installed under ``<CANN>/nnal`` or as a sibling directory;
        an empty ``nnal`` folder is not an installation.
        """
        return nnal_dirs(sdk, leaf)
