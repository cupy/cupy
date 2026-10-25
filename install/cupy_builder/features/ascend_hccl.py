"""HCCL (multi-NPU collective communication) feature for the Ascend backend.

CuPy's ``cupyx.distributed`` is written against NCCL: it calls
``cupy.cuda.nccl`` (``cupy_backends.cuda.libs.nccl``), which is the Cython
binding of ``cupy/backends/cuda/libs/nccl.pyx``.  On Ascend that binding is
compiled unchanged, but ``cupy/backends/cupy_xccl.h`` routes the NCCL API to
``cupy/backends/ascend/cupy_hccl.h`` -- a NCCL-compatible shim implemented on
top of HCCL, exactly like ``hip/cupy_rccl.h`` routes it to RCCL.

This feature owns that one module:

    cupy.backends.cuda.libs.nccl   <-  nccl.pyx + ascend/cupy_hccl.h

Because it is the only thing that needs HCCL, it is **optional** (``required =
False``): a CANN install without ``hccl/hccl.h`` simply produces a CuPy without
``cupy.cuda.nccl``, and ``cupyx.distributed.init_process_group`` then fails with
its own "NCCL is not available" error instead of the build breaking.

Library naming
--------------
* ``libhccl.so``            -- the collective/P2P entry points
                               (``HcclAllReduce``/``HcclSend``/...).  Present in
                               every CANN release.
* communicator management    -- ``HcclGetRootInfo``/``HcclCommInitRootInfo``/
  (``libhcomm.so`` in 9.x,     ``HcclGetRankSize``/``HcclCommDestroy``/... live
   ``libhccl_fwk.so`` in      in the "framework" library, which CANN renamed
   in 8.5, they are the        between releases (9.0 also makes
   same file)                  ``libhccl_fwk.so`` a symlink to ``libhcomm.so``).
* ``libhccl_alg.so`` /      -- 9.x split out libraries that the framework
  ``libhccl_plf.so``           library's *own* undefined references point at;
                               they must be on the link line or ``ld`` reports
                               ``hccl::CollAlgOperator::GetBlockDim`` & friends
                               as undefined.  8.5 needs the same set.

The list is resolved against ``<sdk>/lib64`` and filtered, so a future
re-organisation degrades to "module not built" with a build-time warning rather
than a hard link error.

Set ``CUPY_ENABLE_HCCL=0`` to skip detection entirely, or ``CUPY_ENABLE_HCCL=1``
to make a missing HCCL a hard build failure (CI for the distributed tier).
"""

from __future__ import annotations

import os
from typing import Any

import cupy_builder.install_build as build

from cupy_builder.backends.ascend import cann_arch_dir
from cupy_builder.features._base import Feature


#: The Cython module this feature builds.  The name is NCCL's on purpose: the
#: module path is what ``cupy/cuda/nccl.py`` and ``cupyx/distributed`` import.
HCCL_MODULE = 'cupy.backends.cuda.libs.nccl'

#: Header that must exist for the shim to compile (``<sdk>/include/hccl/hccl.h``).
_HCCL_HEADER = ('hccl', 'hccl.h')

#: Collective/P2P entry points.
_HCCL_OPS_LIB = 'hccl'

#: Communicator management, renamed by CANN (see the module docstring).
_HCCL_FRAMEWORK_LIBS = ('hcomm', 'hccl_fwk')

#: Libraries the framework library itself needs at link time (CANN 8.5 / 9.x).
_HCCL_SUPPORT_LIBS = ('hccl_alg', 'hccl_plf')

#: ACL - used by the shim for ``aclrtMemcpyAsync``/``aclrtGetDevice``.
_HCCL_ACL_LIBS = ('ascendcl',)


def _sdk_lib_dirs(sdk: str) -> list[str]:
    """Directories that may hold the HCCL shared libraries."""
    return [
        os.path.join(sdk, 'lib64'),
        cann_arch_dir(sdk, 'lib64'),
    ]


def _lib_exists(directory: str, name: str) -> bool:
    if os.path.isfile(os.path.join(directory, f'lib{name}.so')):
        return True
    prefix = f'lib{name}.so.'
    try:
        return any(f.startswith(prefix) for f in os.listdir(directory))
    except OSError:
        return False


def has_hccl_header(sdk: str | None) -> bool:
    """Whether ``hccl/hccl.h`` is installed under the CANN tree."""
    if not sdk or sdk == 'NOT_INITIALIZED':
        return False
    for inc in (os.path.join(sdk, 'include'),
                cann_arch_dir(sdk, 'include')):
        if os.path.isfile(os.path.join(inc, *_HCCL_HEADER)):
            return True
    return False


def select_hccl_libraries(sdk: str | None) -> list[str]:
    """Link line for the NCCL binding, filtered against the filesystem.

    Order matters for ``ld``: the framework library's undefined references are
    resolved by the support libraries, which is why the support libraries come
    last.
    """
    libs = [_HCCL_OPS_LIB]

    dirs = [d for d in _sdk_lib_dirs(sdk) if os.path.isdir(d)] if sdk else []
    if not dirs:
        # No filesystem to inspect: propose the full set and let the linker
        # produce the real error instead of silently dropping a library.
        return libs + list(_HCCL_FRAMEWORK_LIBS[:1]) + \
            list(_HCCL_SUPPORT_LIBS) + list(_HCCL_ACL_LIBS)

    def present(name: str) -> bool:
        return any(_lib_exists(d, name) for d in dirs)

    # Exactly one framework library exists per CANN release; prefer the 9.x
    # name when both are present (they are the same file there).
    for name in _HCCL_FRAMEWORK_LIBS:
        if present(name):
            libs.append(name)
            break
    libs += [name for name in _HCCL_SUPPORT_LIBS if present(name)]
    libs += [name for name in _HCCL_ACL_LIBS if present(name)]
    return libs


def _env_flag_off() -> bool:
    return os.environ.get('CUPY_ENABLE_HCCL', '').strip() == '0'


def _env_flag_forced_on() -> bool:
    return os.environ.get('CUPY_ENABLE_HCCL', '').strip() == '1'


class CUPY_ascend_hccl(Feature):
    """HCCL binding: ``cupy.backends.cuda.libs.nccl`` (NCCL API over HCCL).

    Compiled only when the HCCL headers are present.  The generic
    compile+link probe in ``preconfigure_modules`` is the final authority: if
    ``hccl/hccl.h`` exists but the libraries cannot be linked, the module is
    skipped with a warning instead of failing the whole build.
    """

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.name = 'ascend_hccl'
        self.required = False
        # `preconfigure_modules` reads get_version() without calling
        # configure(), so provide a value up front instead of the sentinel.
        # `None` (instead of the 0 used by the sibling features) keeps the
        # summary line a plain "ascend_hccl: Yes": the binding has no version of
        # its own, it tracks the emulated NCCL level reported by
        # ``cupy.cuda.nccl.get_build_version()`` (see cupy_hccl.h).
        self._version = None
        # `<sdk>/include` is already on the include path (see
        # AscendBackend.get_include_dirs); re-checking it here is what makes the
        # availability probe (`check_library(includes=...)`) meaningful.
        self.includes = ['/'.join(_HCCL_HEADER)]

        sdk = build.get_cann_path()
        available = has_hccl_header(sdk)
        libraries = select_hccl_libraries(sdk)

        if _env_flag_off():
            self.modules = []
            self.libraries = []
            print('HCCL detection disabled by CUPY_ENABLE_HCCL=0: '
                  'cupy.cuda.nccl / cupyx.distributed will be unavailable.')
        elif not available and _env_flag_forced_on():
            raise RuntimeError(
                'CUPY_ENABLE_HCCL=1 but hccl/hccl.h was not found under '
                f'{sdk!r}. Install the CANN toolkit (HCCL ships with it) or '
                'unset CUPY_ENABLE_HCCL.')
        elif available:
            self.modules = [HCCL_MODULE]
            self.libraries = libraries
            print('HCCL found: building %s (libraries: %s)'
                  % (HCCL_MODULE, ', '.join(libraries)))
        else:
            # Keep the dependency declared with no module requested so the
            # generic probe reports `ascend_hccl: No` (the "optional module
            # skipped" path) instead of silently passing.
            self.modules = []
            self.libraries = [libraries[0]]
            print('HCCL (hccl/hccl.h) is not installed: the NCCL binding '
                  '(cupy.backends.cuda.libs.nccl) will be skipped, so '
                  'cupy.cuda.nccl.available is False and '
                  'cupyx.distributed.init_process_group raises '
                  '"NCCL is not available". Set CUPY_ENABLE_HCCL=1 to make '
                  'this fatal.')

    def configure(self, compiler: Any, settings: Any) -> bool:
        self._version = None
        return True
