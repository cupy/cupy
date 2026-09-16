"""Optional ops-fft (CANN FFT) feature for the Ascend backend.

**ops-fft is not part of the base CANN SDK.**  ``ascendcl``/``runtime``/``opapi``
always ship with the toolkit, but the FFT operator library is a separate Huawei
package: it is distributed as a ``.run`` file (e.g.
``cann-910b-ops-fft_9.0.0_linux-x86_64.run``) that installs
``libcann_ops_fft.so`` into the CANN tree, and it can also be built from source
(the in-tree build produces the ``.so`` under ``build/``).  A perfectly valid
CANN installation therefore usually has **no** FFT support.

Because of that this feature is **optional** and degrades in two independent
places:

1. *build time* -- when the library cannot be found, the feature contributes no
   modules and the build proceeds without FFT support.  Even when a directory is
   found, the generic availability test in ``cupy_setup_build.py`` still runs a
   compile+**link** probe (``check_library(libraries=['cann_ops_fft'])``); if
   that fails the module is skipped with a warning rather than failing the
   build.
2. *run time* -- ``import cupy``/``import cupy.fft`` never fail.  The binding is
   resolved lazily by :func:`cupy.fft._backend.get_cufft`, which raises a clear
   ``RuntimeError`` explaining that ops-fft was not available at build time.
   Note that a *built* binding still needs ``libcann_ops_fft.so`` on the system
   loader path at run time (the CANN path is deliberately not baked into the
   extension's rpath), otherwise the import fails and the same
   ``RuntimeError`` is produced.

Set ``CUPY_ENABLE_ACLFFT=1`` to turn "not found" into a hard build failure, or
``CUPY_ENABLE_ACLFFT=0`` to skip detection entirely.

Detection order for the explicit ``-L`` directory:

1. ``ASCEND_OPS_FFT_PATH`` environment variable (``<path>``, ``<path>/lib``,
   ``<path>/lib64`` are probed);
2. the CANN tree (``<CANN>/lib64``, ``<CANN>/lib``,
   ``<CANN>/x86_64-linux/lib64``, ``<CANN>/../ops_fft/lib64``) -- where the
   official ``.run`` package installs;
3. a source checkout at ``~/repos/ops-fft/build`` (development fallback);
4. the default linker search path (``ldconfig``), in which case no explicit
   ``-L`` directory is needed but the library is still available -- see
   :func:`has_ops_fft`.

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


def _candidate_dirs() -> list[str]:
    """Directories that may hold ``libcann_ops_fft.so``, in preference order."""
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

    return list(dict.fromkeys(roots))


def _scan_for_lib() -> str | None:
    """First candidate directory that actually contains the library."""
    for root in _candidate_dirs():
        if os.path.isfile(os.path.join(root, _LIB_BASENAME)):
            return root
    return None


def find_ops_fft_lib() -> str | None:
    """Return the directory that must be added as ``-L``, or ``None``.

    ``None`` means "no explicit ``-L`` is needed", which is *not* the same as
    "not available": either the library was not found at all, or it is already
    reachable through the default linker search path.  Use
    :func:`has_ops_fft` to answer "is ops-fft available?"; this function only
    answers "where is its library directory?".

    (Conflating the two used to make the feature skip a perfectly linkable
    library, and made ``CUPY_ENABLE_ACLFFT=1`` fail on a machine where ops-fft
    *was* installed.)
    """
    if _env_flag_off():
        return None
    lib_dir = _scan_for_lib()
    if lib_dir is not None:
        return lib_dir
    # Reachable via the default search path only -> no -L directory to report.
    return None


def has_ops_fft() -> bool:
    """Whether a usable ``libcann_ops_fft.so`` is present on this machine.

    Unlike :func:`find_ops_fft_lib` this also returns True when the library is
    only reachable through the default linker search path (``ldconfig``).
    Whether it can *really* be linked is confirmed afterwards by the generic
    ``check_library(libraries=...)`` probe in ``cupy_setup_build.py``.
    """
    if _env_flag_off():
        return False
    if _scan_for_lib() is not None:
        return True
    # ops-fft is NOT bundled with the base CANN SDK, so a directory scan is not
    # conclusive; the last chance is the system loader path.
    return bool(ctypes.util.find_library('cann_ops_fft'))


def _env_flag_off() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLFFT', '').strip() == '0'


def _env_flag_forced_on() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLFFT', '').strip() == '1'


class CUPY_ascend_fft(Feature):
    """ops-fft (aclfft) binding: ``cupy.backends.ascend.api.aclfft``.

    Compiled only when ``libcann_ops_fft.so`` is available; otherwise the
    feature degrades to an empty module list so the rest of the Ascend build is
    unaffected (see the module docstring for the full degradation story).
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

        available = has_ops_fft()
        lib_dir = find_ops_fft_lib()

        if _env_flag_off():
            # CUPY_ENABLE_ACLFFT=0: skip detection entirely.
            self.modules = []
            self.libraries = []
            print('ops-fft (CANN FFT) detection disabled by '
                  'CUPY_ENABLE_ACLFFT=0.')
        elif not available and _env_flag_forced_on():
            raise RuntimeError(
                'CUPY_ENABLE_ACLFFT=1 but libcann_ops_fft.so was not found; '
                'set ASCEND_OPS_FFT_PATH or install ops-fft (see '
                'docs/ascend/ascend_fft.md).')
        elif available:
            self.modules = ['cupy.backends.ascend.api.aclfft']
            self.libraries = ['cann_ops_fft']
            print('ops-fft (CANN FFT) found: %s'
                  % (lib_dir or 'default linker search path'))
        else:
            # Not installed.  Keep the *dependency* declared even though no
            # module is requested: `preconfigure_modules` runs a compile+link
            # probe with `libraries`, which is what turns "ops-fft is absent"
            # into the standard "optional module skipped" path (summary line
            # `ascend_fft: No`).  Reporting it as available with an empty
            # library list would make the probe trivially pass.
            self.modules = []
            self.libraries = ['cann_ops_fft']
            print('ops-fft (CANN FFT) is not installed (it is not part of the '
                  'base CANN SDK): the ascend_fft module will be skipped and '
                  'cupy.fft calls will raise RuntimeError. Set '
                  'ASCEND_OPS_FFT_PATH to point at it, or CUPY_ENABLE_ACLFFT=1 '
                  'to make its absence fatal.')

        # keep the resolved dir for backends/ascend.py to pick up; None means
        # "nothing to add to the linker search path".
        self._lib_dir = lib_dir

    def get_lib_dir(self) -> str | None:
        return self._lib_dir

    def configure(self, compiler: Any, settings: Any) -> bool:
        self._version = 0
        return True
