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
import importlib.util
import os
import platform
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build
import cupy_builder.install_utils as utils

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


# ---------------------------------------------------------------------------
# AOT-compiled AscendC kernel fatbins (see docs/ascend/Package.md §2.10)
# ---------------------------------------------------------------------------
#: SoCs to compile the built-in custom kernels for. Comma/space separated.
#: Defaults to ``CUPY_ASCEND_SOC`` -- the same value the runtime uses -- so the
#: fatbins in the wheel are the ones a default install looks for.
KERNEL_SOCS_ENV_VAR = 'CUPY_ASCEND_KERNEL_SOCS'

#: Set to ``0`` to ship kernel *sources* only: the installed wheel then
#: JIT-compiles them on first import (bisheng is present on every CANN box).
AOT_KERNELS_ENV_VAR = 'CUPY_ASCEND_AOT_KERNELS'

#: Fatbin sub-directory of the kernel source directory. Keep in sync with
#: ``kernels.AOT_DIR_NAME`` in ``cupy/backends/ascend/kernels/__init__.py``.
_KERNEL_AOT_DIR = '_aot'

#: Where the built-in AscendC kernels and their registry live.
_KERNEL_SRC_SUBDIR = os.path.join('cupy', 'backends', 'ascend', 'kernels')

#: bisheng wrapper, shared with the runtime JIT path (loaded by path).
_BISHENG_MODULE_PATH = os.path.join('cupy', 'backends', 'ascend', 'bisheng.py')


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean ``CUPY_*`` variable (``0``/``false``/``no``/``off`` = off)."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off', '')


def _kernel_socs(bisheng: Any) -> list[str]:
    """SoCs to AOT-compile the built-in kernels for.

    Falls back to :func:`bisheng.default_soc` (``CUPY_ASCEND_SOC``, default
    ``Ascend910B4``) rather than inventing a list: the aicore ISA is
    SoC-specific, so a fatbin is only useful if its SoC name matches what the
    runtime asks for.
    """
    raw = os.environ.get(KERNEL_SOCS_ENV_VAR, '')
    if not raw.strip():
        return [bisheng.default_soc()]
    for sep in (';', ' ', '\t', '\n'):
        raw = raw.replace(sep, ',')
    return [soc.strip() for soc in raw.split(',') if soc.strip()]


def _load_bisheng(source_root: str) -> Any:
    """Import ``cupy/backends/ascend/bisheng.py`` without importing ``cupy``.

    ``setup.py`` can not import the package it is building (``cupy/__init__.py``
    does far more than expose this module, and it needs the extensions being
    built), while ``bisheng.py`` itself needs only the standard library. Loading
    it by path keeps a single source of truth for the bisheng command line and
    the AscendC include directory discovery.
    """
    path = os.path.join(source_root, _BISHENG_MODULE_PATH)
    spec = importlib.util.spec_from_file_location('_cupy_ascend_bisheng', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    #: Headers of the stateless ``aclnn_rand`` op family backing cupy.random.
    #: All three must be installed for the random ops to be compiled in.
    _ACLRAND_HEADERS = ('aclnn_uniform.h', 'aclnn_normal.h', 'aclnn_random.h')

    def has_aclnn_rand(self) -> bool:
        """Feature-detect the stateless aclnn rand ops used by cupy.random.

        The ``aclnn_rand`` family (``aclnnInplaceUniform/Normal/Random``) is
        NOT available in every CANN release / SoC package, so it must not be
        compiled unconditionally (docs/ascend/DeveloperNotes.md §ops-rand).

        Detection is header-presence based, mirroring the two include roots
        actually used at compile time (``<sdk>/include`` and the arch-specific
        layout). ``CUPY_ENABLE_ACLRAND=0`` forces it off, ``=1`` skips the
        header check (cross builds without the SDK at hand).

        The result feeds BOTH conditional-compilation channels (see §3.5 of
        DeveloperNotes.md): the C macro ``CUPY_CANN_HAS_RAND`` (``#if`` in
        acl_random_ops.h) and the Cython compile-time constant of the same
        name (``IF`` in acl_utils.pyx).
        """
        env = os.environ.get('CUPY_ENABLE_ACLRAND')
        if env == '0':
            return False
        if env == '1':
            return True
        sdk = self.get_sdk_path()
        if not sdk or sdk == 'NOT_INITIALIZED':
            return False
        for header in self._ACLRAND_HEADERS:
            rel = os.path.join('aclnnop', header)
            if os.path.isfile(os.path.join(sdk, 'include', rel)):
                continue
            arch_include = os.path.join(cann_arch_dir(sdk, 'include'), rel)
            if os.path.isfile(arch_include):
                continue
            return False
        return True

    def get_define_macros(self) -> list[tuple[str, str]]:
        return [
            ('CUPY_USE_ASCEND', '1'),
            # keep in sync with the Cython compile-time constant
            ('CUPY_CANN_VERSION', str(self.get_version())),
            # keep in sync with the Cython compile-time constant
            ('CUPY_CANN_HAS_RAND', str(int(self.has_aclnn_rand()))),
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
            # keep in sync with the C macro of the same name; gates the
            # `IF` blocks for cupy.random in acl_utils.pyx
            'CUPY_CANN_HAS_RAND': int(self.has_aclnn_rand()),
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
    # AOT-compiled AscendC kernel fatbins
    # ------------------------------------------------------------------
    def prebuild_artifacts(self, ctx: Context) -> list[str]:
        """AOT-compile the built-in AscendC kernels for the wheel.

        Those kernels are normally JIT-compiled on first import, because a
        fatbin is SoC-specific and a build machine usually has no NPU to probe
        the target SoC from. Compiling for an explicit SoC list here means the
        installed wheel only needs to *load* a fatbin, for the SoCs it was built
        for; any other SoC still falls back to JIT at runtime (see
        ``kernels.ensure_built``).

        SoCs come from ``CUPY_ASCEND_KERNEL_SOCS`` (default ``CUPY_ASCEND_SOC``,
        i.e. ``Ascend910B4``); ``CUPY_ASCEND_AOT_KERNELS=0`` ships sources only.
        bisheng cross-compiles from any host, so this needs no NPU -- but it
        does need the AscendC headers of the local CANN install.

        Never raises: a wheel without prebuilt fatbins is still functional, it
        just pays the JIT cost for every user.
        """
        if not _env_flag(AOT_KERNELS_ENV_VAR, default=True):
            print(f'Ascend: AOT kernels disabled ({AOT_KERNELS_ENV_VAR}=0); '
                  'the wheel ships sources only and JIT-compiles on first use')
            return []

        kernels_dir = os.path.join(ctx.source_root, _KERNEL_SRC_SUBDIR)
        srcs = sorted(glob.glob(os.path.join(kernels_dir, '*.cpp')))
        if not srcs:
            print(f'Ascend: no kernel sources under {kernels_dir}, '
                  'nothing to prebuild')
            return []

        dest = os.path.join(kernels_dir, _KERNEL_AOT_DIR)
        try:
            bisheng = _load_bisheng(ctx.source_root)
            socs = _kernel_socs(bisheng)
            print(f'Ascend: AOT-compiling {len(srcs)} kernel source(s) for '
                  f'{", ".join(socs)} ...')
            built = bisheng.prebuild(srcs, dest, socs)
        except Exception as e:
            utils.print_warning(
                f'could not prebuild the AscendC kernel fatbins ({e})',
                'the wheel will JIT-compile them on first import, which needs '
                'bisheng on the target machine',
                f'to build for specific SoC(s), set {KERNEL_SOCS_ENV_VAR}')
            return []

        artifacts = [path for paths in built.values() for path in paths]
        print(f'Ascend: prebuilt kernels ready: {len(artifacts)} fatbin(s) '
              f'under {dest}')
        return artifacts

    # ------------------------------------------------------------------
    @staticmethod
    def _nnal_dirs(sdk: str, leaf: str) -> list[str]:
        """NNAL include/lib dirs of a *verified* install (see ``nnal_dirs``).

        NNAL is installed under ``<CANN>/nnal`` or as a sibling directory;
        an empty ``nnal`` folder is not an installation.
        """
        return nnal_dirs(sdk, leaf)
