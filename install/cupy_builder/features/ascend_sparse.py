"""Optional ops-sparse (CANN sparse) feature for the Ascend backend.

**ops-sparse is not part of the base CANN SDK.**  ``ascendcl``/``runtime``/
``opapi``/``nnopbase`` always ship with the toolkit, but the sparse operator
library is a separate Huawei package: it is distributed as a ``.run`` file
(e.g. ``cann-950-ops-sparse-1.0.0_linux-x86_64.run``) that installs
``libops_sparse.so`` into the CANN tree, and it can also be built from source
(the in-tree build produces the ``.so`` under ``build/``).  A perfectly valid
CANN installation therefore usually has **no** sparse support.

Because of that this feature is **optional** and degrades in two independent
places:

1. *build time* -- when the library cannot be found, the feature contributes no
   modules and the build proceeds without sparse support.  Even when a
   directory is found, the generic availability test in
   ``cupy_setup_build.py`` still runs a compile+**link** probe
   (``check_library(libraries=['ops_sparse'])``); if that fails the module is
   skipped with a warning rather than failing the build.
2. *run time* -- ``import cupy``/``import cupyx.cusparse`` never fail.  The
   binding is resolved lazily by :func:`cupyx._cusparse_backend.get_cusparse`,
   which raises a clear ``RuntimeError`` explaining that ops-sparse was not
   available at build time.  Note that a *built* binding still needs
   ``libops_sparse.so`` on the system loader path at run time (the CANN path is
   deliberately not baked into the extension's rpath), otherwise the import
   fails and the same ``RuntimeError`` is produced.

Set ``CUPY_ENABLE_ACLSPARSE=1`` to turn "not found" into a hard build failure,
or ``CUPY_ENABLE_ACLSPARSE=0`` to skip detection entirely.

Detection order for the explicit ``-L`` directory:

1. ``ASCEND_OPS_SPARSE_PATH`` environment variable (``<path>``, ``<path>/lib``,
   ``<path>/lib64`` are probed);
2. the CANN tree -- where the official ``.run`` package installs.  The
   installer appends the toolkit directory to the given ``--install-path``, so
   two layouts are probed: ``<CANN>/lib64`` (installed with
   ``--install-path="$ASCEND_HOME_PATH/.."``, the shared base-operator layout
   used by the other ``ops-*`` packages) and ``<CANN>/cann/lib64`` (installed
   with ``--install-path="$ASCEND_HOME_PATH"``, which adds an extra ``cann``
   level), each also under ``lib/`` and ``x86_64-linux/lib64/``, plus
   ``<CANN>/../ops_sparse/lib64``;
3. the default linker search path (``ldconfig``), in which case no explicit
   ``-L`` directory is needed but the library is still available -- see
   :func:`has_ops_sparse`.

The public header ``cann_ops_sparse.h`` is vendored next to the binding module
(``cupy/backends/ascend/api/cann_ops_sparse.h``), so only the *library* needs to
be present at build time.
"""
from __future__ import annotations

import ctypes.util
import os
from typing import Any

import cupy_builder.install_build as build

from cupy_builder.features._base import Feature

# The installed library is named ``libops_sparse.so`` (both the official .run
# package and the in-source build produce this name); ``libcann_ops_sparse.so``
# is accepted as an alias for older/newer packagings.
_LIB_BASENAMES = ('libops_sparse.so', 'libcann_ops_sparse.so')


def _candidate_dirs() -> list[str]:
    """Directories that may hold ``libops_sparse.so``, in preference order."""
    roots: list[str] = []

    # 1) explicit override
    override = os.environ.get('ASCEND_OPS_SPARSE_PATH')
    if override:
        roots += [override, os.path.join(override, 'lib'),
                  os.path.join(override, 'lib64')]

    # 2) CANN tree (the .run package's install location).  The installer
    #    appends the toolkit dir to the given --install-path:
    #      --install-path="$ASCEND_HOME_PATH/.." -> <CANN>/lib64
    #          (shared base-operator layout, same as the other ops-* packages)
    #      --install-path="$ASCEND_HOME_PATH"    -> <CANN>/cann/lib64
    #          (one extra "cann" level, older/alternative install)
    cann_path = build.get_cann_path()
    if cann_path and cann_path != 'NOT_INITIALIZED':
        for prefix in (cann_path, os.path.join(cann_path, 'cann')):
            roots += [
                os.path.join(prefix, 'lib64'),
                os.path.join(prefix, 'lib'),
                os.path.join(prefix, 'x86_64-linux', 'lib64'),
            ]
        roots.append(
            os.path.join(os.path.dirname(cann_path), 'ops_sparse', 'lib64'))

    return list(dict.fromkeys(roots))


def _scan_for_lib() -> str | None:
    """First candidate directory that actually contains the library."""
    for root in _candidate_dirs():
        for basename in _LIB_BASENAMES:
            if os.path.isfile(os.path.join(root, basename)):
                return root
    return None


def find_ops_sparse_lib() -> str | None:
    """Return the directory that must be added as ``-L``, or ``None``.

    ``None`` means "no explicit ``-L`` is needed", which is *not* the same as
    "not available": either the library was not found at all, or it is already
    reachable through the default linker search path.  Use
    :func:`has_ops_sparse` to answer "is ops-sparse available?"; this function
    only answers "where is its library directory?".
    """
    if _env_flag_off():
        return None
    lib_dir = _scan_for_lib()
    if lib_dir is not None:
        return lib_dir
    # Reachable via the default search path only -> no -L directory to report.
    return None


def has_ops_sparse() -> bool:
    """Whether a usable ``libops_sparse.so`` is present on this machine.

    Unlike :func:`find_ops_sparse_lib` this also returns True when the library
    is only reachable through the default linker search path (``ldconfig``).
    Whether it can *really* be linked is confirmed afterwards by the generic
    ``check_library(libraries=...)`` probe in ``cupy_setup_build.py``.
    """
    if _env_flag_off():
        return False
    if _scan_for_lib() is not None:
        return True
    # ops-sparse is NOT bundled with the base CANN SDK, so a directory scan is
    # not conclusive; the last chance is the system loader path.
    for basename in _LIB_BASENAMES:
        if ctypes.util.find_library(basename[3:-3]):
            return True
    return False


def _env_flag_off() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLSPARSE', '').strip() == '0'


def _env_flag_forced_on() -> bool:
    return os.environ.get('CUPY_ENABLE_ACLSPARSE', '').strip() == '1'


class CUPY_ascend_sparse(Feature):
    """ops-sparse (aclsparse) binding: ``cupy.backends.ascend.api.acl_sparse``.

    Compiled only when ``libops_sparse.so`` is available; otherwise the
    feature degrades to an empty module list so the rest of the Ascend build
    is unaffected (see the module docstring for the full degradation story).
    """

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.name = 'ascend_sparse'
        self.required = False
        # `preconfigure_modules` reads get_version() without calling
        # configure() (that call is commented out there), so provide a value
        # up front instead of the `_UNDETERMINED` sentinel.
        self._version = 0
        # The header is vendored next to acl_sparse.pyx, so no include dirs
        # here.
        self.includes = []

        available = has_ops_sparse()
        lib_dir = find_ops_sparse_lib()

        if _env_flag_off():
            # CUPY_ENABLE_ACLSPARSE=0: skip detection entirely.
            self.modules = []
            self.libraries = []
            print('ops-sparse (CANN sparse) detection disabled by '
                  'CUPY_ENABLE_ACLSPARSE=0.')
        elif not available and _env_flag_forced_on():
            raise RuntimeError(
                'CUPY_ENABLE_ACLSPARSE=1 but libops_sparse.so was not found; '
                'set ASCEND_OPS_SPARSE_PATH or install ops-sparse (see '
                'docs/ascend/ascend_sparse.md).')
        elif available:
            self.modules = ['cupy.backends.ascend.api.acl_sparse']
            self.libraries = ['ops_sparse']
            print('ops-sparse (CANN sparse) found: %s'
                  % (lib_dir or 'default linker search path'))
        else:
            # Not installed.  Keep the *dependency* declared even though no
            # module is requested: `preconfigure_modules` runs a compile+link
            # probe with `libraries`, which is what turns "ops-sparse is absent"
            # into the standard "optional module skipped" path (summary line
            # `ascend_sparse: No`).  Reporting it as available with an empty
            # library list would make the probe trivially pass.
            self.modules = []
            self.libraries = ['ops_sparse']
            print('ops-sparse (CANN sparse) is not installed (it is not part '
                  'of the base CANN SDK and requires CANN >= 9.0): the '
                  'ascend_sparse module will be skipped and cupyx.cusparse '
                  'calls will raise RuntimeError. Set ASCEND_OPS_SPARSE_PATH '
                  'to point at it, or CUPY_ENABLE_ACLSPARSE=1 to make its '
                  'absence fatal.')

        # keep the resolved dir for backends/ascend.py to pick up; None means
        # "nothing to add to the linker search path".
        self._lib_dir = lib_dir

    def get_lib_dir(self) -> str | None:
        return self._lib_dir

    def configure(self, compiler: Any, settings: Any) -> bool:
        self._version = 0
        return True
