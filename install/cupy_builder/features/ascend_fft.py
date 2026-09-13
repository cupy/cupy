"""Optional ops-fft (CANN FFT) feature for the Ascend backend.

ops-fft is Huawei's FFT operator library for CANN. It is distributed as a
``.run`` package (e.g. ``cann-910b-ops-fft_9.0.0_linux-x86_64.run``) that
installs ``libcann_ops_fft.so`` into the CANN tree, and can also be built from
source (the in-tree build produces the ``.so`` under ``build/``).

This feature is **optional**: when the library cannot be found, the feature
silently contributes no modules and the build proceeds without FFT support
(``import cupy.fft`` still works, calling an FFT function raises a clear
``RuntimeError`` from :func:`cupy.fft._backend.get_cufft`). Set
``CUPY_ENABLE_ACLFFT=1`` to fail the build instead when the library is
missing, or ``CUPY_ENABLE_ACLFFT=0`` to skip detection entirely.

Detection order for the library directory:

1. ``ASCEND_OPS_FFT_PATH`` environment variable (``<path>``, ``<path>/lib``,
   ``<path>/lib64`` are probed);
2. the CANN tree (``<CANN>/lib64``, ``<CANN>/lib``,
   ``<CANN>/x86_64-linux/lib64``, ``<CANN>/../ops_fft/lib64``) -- where the
   official ``.run`` package installs;
3. a source checkout at ``~/repos/ops-fft/build`` (development fallback);
4. the default linker search path (``ldconfig``), in which case no explicit
   ``-L`` directory is needed.

The public header ``cann_ops_fft.h`` is vendored next to the binding module
(``cupy/backends/ascend/api/cann_ops_fft.h``), so only the *library* needs to
be present at build time.
"""
from __future__ import annotations

import ctypes.util
import os
from typing import Any

import cupy_builder.install_build as build

from cupy_builder.features._base import Feature

_LIB_BASENAME = 'libcann_ops_fft.so'


def find_ops_fft_lib() -> str | None:
    """Return the directory containing ``libcann_ops_fft.so``, or ``None``.

    ``None`` means the library was not found in any known location; if it is
    still reachable via the default linker search path, linking with
    ``-lcann_ops_fft`` may still succeed (checked by the build system).
    """
    if _env_flag_off():
        return None

    roots: list[str] = []

    # 1) explicit override
    override = os.environ.get('ASCEND_OPS_FFT_PATH')
    if override:
        roots += [override, os.path.join(override, 'lib'),
                  os.path.join(override, 'lib64')]

    # 2) CANN tree (the .run package's default install location)
    cann_path = build.get_cann_path()
    if cann_path and cann_path != 'NOT_INITIALIZED':
        roots += [
            os.path.join(cann_path, 'lib64'),
            os.path.join(cann_path, 'lib'),
            os.path.join(cann_path, 'x86_64-linux', 'lib64'),
            os.path.join(os.path.dirname(cann_path), 'ops_fft', 'lib64'),
        ]

    # 3) development source checkout
    dev_root = os.path.expanduser('~/repos/ops-fft')
    roots += [os.path.join(dev_root, 'build'), os.path.join(dev_root, 'lib64')]

    for root in roots:
        if os.path.isfile(os.path.join(root, _LIB_BASENAME)):
            return root

    # 4) default linker search path
    if ctypes.util.find_library('cann_ops_fft'):
        return None
    return None


def _env_flag_off() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLFFT', '').strip() == '0'


def _env_flag_forced_on() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLFFT', '').strip() == '1'


class CUPY_ascend_fft(Feature):
    """ops-fft (aclfft) binding: ``cupy.backends.ascend.api.aclfft``.

    Compiled only when ``libcann_ops_fft.so`` is detected; otherwise the
    feature degrades to an empty module list so the rest of the Ascend build
    is unaffected.
    """

    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.name = 'ascend_fft'
        self.required = False
        # `preconfigure_modules` reads get_version() without calling
        # configure() (that call is commented out there), so provide a value
        # up front instead of the `_UNDETERMINED` sentinel.
        self._version = 0
        # The header is vendored next to aclfft.pyx, so no include dirs here.
        self.includes = []

        lib_dir = find_ops_fft_lib()
        if lib_dir is None and _env_flag_forced_on():
            raise RuntimeError(
                'CUPY_ENABLE_ACLFFT=1 but libcann_ops_fft.so was not found; '
                'set ASCEND_OPS_FFT_PATH or install ops-fft (see '
                'docs/ascend/ascend_fft.md).')

        if lib_dir is not None:
            self.modules = ['cupy.backends.ascend.api.aclfft']
            self.libraries = ['cann_ops_fft']
            print(f'ops-fft (CANN FFT) found at: {lib_dir}')
        else:
            self.modules = []
            self.libraries = []
            print('ops-fft (CANN FFT) not found: building without FFT '
                  'support (cupy.fft calls will raise at runtime).')

        # keep the resolved dir for backends/ascend.py to pick up
        self._lib_dir = lib_dir

    def get_lib_dir(self) -> str | None:
        return self._lib_dir

    def configure(self, compiler: Any, settings: Any) -> bool:
        self._version = 0
        return True
