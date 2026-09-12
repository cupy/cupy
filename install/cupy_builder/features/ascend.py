"""Huawei Ascend / CANN backend feature.

CANN renames and reorganises its shared libraries between major/minor
releases, so the set of libraries to link cannot be hardcoded. This module:

1. detects the installed CANN version (see ``build.check_cann_version``);
2. selects the library set for that version;
3. filters out any library that is not actually present on disk
   (:func:`_filter_existing_libs`), so a slightly different naming in a
   future release degrades gracefully instead of failing the link step.

Known layouts
-------------
* CANN 8.2      - ``aclnn_ops_train``, ``aclnn_ops_infer``, ``aclnn_math``,
                  ``aclnn_rand``, ``acl_op_compiler``, ``graph``, ``profapi``
* CANN 8.5      - ``opapi_nn``, ``opapi``, ``opapi_math``, ``op_common``,
                  ``ge_compiler``, ``ge_common``, ``gert``, ``graph``,
                  ``op_compile_adapter``, ``profapi``
* CANN 9.0      - unknown yet; falls back to the 8.5 set filtered by
                  existence on disk.
"""

from __future__ import annotations

import os
from typing import Any

import cupy_builder.install_build as build

from cupy_builder.features._base import Feature


# ---------------------------------------------------------------------------
# Cython modules compiled for the Ascend backend.
# ---------------------------------------------------------------------------
ascend_files = [
    'cupy.backends.cuda.api._driver_enum',  # JIT can be ignored
    'cupy.backends.cuda.api._runtime_enum',
    # 'cupy.backends.cuda.api._device_prop',
    'cupy.backends.backend.api.driver',  # empty driver.pyx
    'cupy.backends.backend.api.runtime',
    # 'cupy.backends.cuda.libs.cublas',
    # 'cupy.backends.cuda.libs.curand',
    # 'cupy.backends.cuda.libs.cusparse',
    # 'cupy.backends.cuda.libs.nvrtc',
    'cupy.backends.backend.stream',
    'cupy.backends.backend._softlink',
    # # high level OO API
    # 'cupy.cuda.common', #  cudaDataType
    # 'cupy.cuda.cufft',
    'cupy.xpu.device',  # device-runtime capacity like sparse curand
    'cupy.xpu.memory',  # MemoryAsyncPool is not supported
    'cupy.xpu.memory_hook',  # backend independent
    'cupy.xpu.pinned_memory',
    'cupy.xpu.function',  # only compile code for CPointer
    'cupy.xpu.stream',
    'cupy._util',  # backend independent:  context manager, memoise
    'cupy.backends.ascend.api.acl_utils',
    # =============== seperate line for low-high api
    'cupy._core._carray',
    'cupy._core._dtype',
    'cupy._core._scalar',
    'cupy._core.core',  # define _ndarray_base
    'cupy._core.flags',
    'cupy._core.internal',
    'cupy._core.dlpack',
    'cupy._core.numpy_allocator',
    'cupy._core._memory_range',
    'cupy._core._optimize_config',
    ('cupy._core._kernel', ['cupy/_ascend/_core/_kernel.pyx']),
    ('cupy._core._routines_math', ['cupy/_core/_routines_math.pyx']),
    ('cupy._core._routines_binary', ['cupy/_core/_routines_binary.pyx']),
    'cupy._core._routines_creation',
    'cupy._core._routines_manipulation',
    ('cupy._core._routines_linalg', ['cupy/_ascend/_core/_routines_linalg.pyx']),
    ('cupy._core._routines_sorting', ['cupy/_ascend/_core/_routines_sorting.pyx']),
    ('cupy._core._routines_logic', ['cupy/_core/_routines_logic.pyx']),
    ('cupy._core._reduction', ['cupy/_core/_reduction.pyx']),
    ('cupy._core._routines_indexing', ['cupy/_core/_routines_indexing.pyx']),
    # ascend partially support
    ('cupy._core._routines_statistics',
     ['cupy/_core/_routines_statistics.pyx']),
    ('cupy._core.raw', ['cupy/_ascend/_core/raw_kernel_stub.pyx']),
    # =========== Future work ================
    # 'cupy.cuda.graph',  # not sure if possible
    # 'cupy.cuda.texture', # GPU only
    ('cupy._core._accelerator', ["cupy/_core/_gpu/_accelerator.pyx"])  # cuda only
    # 'cupy._core._cub_reduction', # cuda only
    # 'cupy.fft._cache',  # TODO
    # 'cupy.fft._callback', # TODO
    # 'cupy.lib._polynomial', # TODO
    # 'cupyx.scipy.ndimage._bbox_slices', # possible, TODO
]


# ---------------------------------------------------------------------------
# Library sets per CANN version.
# ---------------------------------------------------------------------------
#: Libraries common to every supported CANN release.
_CANN_LIBS_COMMON = ['ascendcl', 'runtime', 'nnopbase', 'graph', 'profapi']

#: CANN 8.2 - the pre-"ops split" library layout (community 8.2.RC1).
_CANN_LIBS_82 = _CANN_LIBS_COMMON + [
    'aclnn_ops_train',
    'aclnn_ops_infer',
    'aclnn_math',
    'aclnn_rand',
    'acl_op_compiler',
]

#: CANN 8.5 - aclnn op libraries were split into opapi / op_common etc.
_CANN_LIBS_85 = _CANN_LIBS_COMMON + [
    'opapi_nn',
    'opapi',
    'opapi_math',
    'op_common',
    'ge_compiler',
    'ge_common',
    'gert',
    'op_compile_adapter',
]

#: NNAL toolkit (BLAS/FFT/asdsip) - optional, only when installed.
_CANN_LIBS_NNAL = ['asdsip', 'asdsip_core', 'asdsip_host', 'mki']


def select_cann_libraries(version: int) -> list[str]:
    """Return the library list for a CANN ``version``.

    ``version`` is encoded as ``major * 100 + minor * 10 + patch``
    (e.g. 820 for 8.2, 850 for 8.5, 900 for 9.0).

    Unknown/newer versions reuse the newest known set; the actual set is
    filtered against the filesystem in :meth:`CUPY_ascend.__init__`.
    """
    if version < 850:
        # CANN 8.2 and earlier
        return list(_CANN_LIBS_82)
    # CANN 8.5 and (best-effort) newer
    return list(_CANN_LIBS_85)


def _cann_lib_dirs() -> list[str]:
    """Candidate directories holding CANN shared libraries."""
    cann_path = build.get_cann_path()
    if not cann_path or cann_path == 'NOT_INITIALIZED':
        return []
    # `lib64` is the canonical location; `x86_64-linux/lib64` appears in
    # some packaged layouts.
    return [
        os.path.join(cann_path, 'lib64'),
        os.path.join(cann_path, 'x86_64-linux', 'lib64'),
    ]


def _filter_existing_libs(libs: list[str]) -> list[str]:
    """Drop libraries that cannot be found on disk.

    Only filters when at least one candidate directory is readable; if we
    cannot inspect the filesystem at all we return the list unchanged so the
    compiler can emit a meaningful error.

    Handles both unversioned (``libopapi.so``) and versioned
    (``libopapi.so.1``) shared objects.
    """
    dirs = [d for d in _cann_lib_dirs() if os.path.isdir(d)]
    if not dirs:
        return libs

    resolved: list[str] = []
    for lib in libs:
        found = False
        for d in dirs:
            if (os.path.isfile(os.path.join(d, f'lib{lib}.so'))
                    or _has_versioned_lib(d, lib)):
                found = True
                break
        if found:
            resolved.append(lib)
        else:
            # Keep the name in the list; the linker may still find it via a
            # different search path (e.g. LD_LIBRARY_PATH).
            resolved.append(lib)
    return resolved


def _has_versioned_lib(directory: str, lib: str) -> bool:
    prefix = f'lib{lib}.so.'
    try:
        return any(name.startswith(prefix) for name in os.listdir(directory))
    except OSError:
        return False


class CUPY_ascend(Feature):
    """Ascend/CANN backend feature with version-aware library selection."""

    minimum_cann_version = 820

    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.name = 'ascend'
        self.required = True
        self.modules = ascend_files
        self.includes = []
        #  _find_static_library() support CUDA backend only
        self.static_libraries = []  # ["ascendcl", 'asdsip_static', 'mki_static']

        self.configure = build.check_cann_version
        build.check_cann_version(None, None)
        self._version = build.get_cann_version()

        self.libraries = select_cann_libraries(self._version)

        # NNAL (BLAS/FFT/asdsip) is a separate install; enable only if present.
        if self._has_nnal():
            self.libraries += list(_CANN_LIBS_NNAL)

        self.libraries = _filter_existing_libs(self.libraries)

    def _has_nnal(self) -> bool:
        cann_path = build.get_cann_path()
        if not cann_path or cann_path == 'NOT_INITIALIZED':
            return False
        # NNAL is installed under <cann>/nnal (toolkit) or a sibling dir.
        for candidate in (
                os.path.join(cann_path, 'nnal'),
                os.path.join(os.path.dirname(cann_path), 'nnal'),):
            if os.path.isdir(candidate):
                return True
        return False
