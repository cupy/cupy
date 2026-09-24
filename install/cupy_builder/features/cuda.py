"""NVIDIA CUDA backend feature.

Defines the ``cuda`` feature itself (:class:`CUDA_cuda`) plus the helper dicts
for the optional CUDA libraries (cuSOLVER, NCCL, NVTX, cuTENSOR, CUB, Jitify,
cuRAND, Thrust, cuSPARSELt) and the shared DLPack feature.
"""

from __future__ import annotations

import sys
from typing import Any

import cupy_builder.install_build as build
import cupy_builder.install_utils as utils
from cupy_builder import Context

from cupy_builder.features._base import Feature


# Libraries required for cudart_static
_cudart_static_libs = (
    ['pthread', 'rt', 'dl'] if sys.platform == 'linux' else []
)


# The value of the key 'file' is a list that contains extension names
# or tuples of an extension name and a list of other sources files
# required to build the extension such as .cpp files and .cu files.
#
#   <extension name> | (<extension name>, a list of <other source>)
#
# The extension name is also interpreted as the name of the Cython
# source file required to build the extension with appending '.pyx'
# file extension.

# Extension name -> sources.  A plain string means the Cython source sits at
# ``<pkg>/<mod>.pyx``; a tuple means "build this module from the sources in the
# list" (see ``cupy_setup_build.module_extension_sources``), which is how the
# backend-specific implementations below are selected.  The two layouts are:
#
#   ``cupy/_core/*.pyx``        backend-neutral / shared (CUDA code paths live
#                               in `IF CUPY_CANN_VERSION <= 0` blocks)
#   ``cupy/_core/_gpu/*.pyx``   CUDA implementation of a shared interface
#                               declared in ``cupy/_core/<name>.pxd``
#
# Ascend counterparts are listed in ``features/ascend.py`` (``ascend_files``).
cuda_files = [
    'cupy.backends.cuda.api._driver_enum',  # JIT can be ignored
    'cupy.backends.cuda.api._runtime_enum',
    'cupy.backends.backend.api.driver',  # empty driver.pyx
    'cupy.backends.backend.api.runtime',
    'cupy.backends.cuda.libs.cublas',
    'cupy.backends.cuda.libs.curand',
    'cupy.backends.cuda.libs.cusparse',
    'cupy.backends.cuda.libs.nvrtc',
    'cupy.backends.backend.stream',
    'cupy.backends.backend._softlink',
    # python cude backend api
    'cupy.cuda.common',
    'cupy.cuda.cufft',
    'cupy.xpu.device',
    'cupy.xpu.memory',
    'cupy.xpu.memory_hook',
    'cupy.xpu.pinned_memory',
    'cupy.xpu.function',
    'cupy.xpu.stream',
    'cupy.xpu.graph',
    'cupy._core._carray',
    'cupy._core._dtype',
    'cupy._core._scalar',
    'cupy._core.core',
    'cupy._core.flags',
    'cupy._core.internal',
    'cupy._core._memory_range',
    'cupy._core._optimize_config',
    ('cupy._core._accelerator', ['cupy/_core/_gpu/_accelerator.pyx']),
    ('cupy._core._fusion_kernel', ['cupy/_core/_gpu/_fusion_kernel.pyx']),
    ('cupy._core._fusion_thread_local',
     ['cupy/_core/_gpu/_fusion_thread_local.pyx']),
    ('cupy._core._fusion_trace', ['cupy/_core/_gpu/_fusion_trace.pyx']),
    ('cupy._core._fusion_variable', ['cupy/_core/_gpu/_fusion_variable.pyx']),
    ('cupy._core.fusion', ['cupy/_core/_gpu/fusion.pyx']),
    ('cupy._core.new_fusion', ['cupy/_core/_gpu/new_fusion.pyx']),
    ('cupy._core._kernel', ['cupy/_core/_gpu/_kernel.pyx']),
    # Canonical module names: `cupy/_core/<name>.pxd` is the shared interface
    # (a copy of the `_gpu/<name>.pxd` one), which is exactly what callers
    # cimport (`from cupy._core._compile_with_cache cimport ...`).
    ('cupy._core._compile_with_cache',
     ['cupy/_core/_gpu/_compile_with_cache.pyx']),
    ('cupy._core._cub_reduction', ['cupy/_core/_gpu/_cub_reduction.pyx']),
    ('cupy._core._reduction', ['cupy/_core/_gpu/_reduction.pyx']),
    'cupy._core._routines_binary',
    ('cupy._core._routines_creation', ['cupy/_core/_routines_creation.pyx']),
    ('cupy._core._routines_indexing', ['cupy/_core/_routines_indexing.pyx']),
    ('cupy._core._routines_linalg', ['cupy/_core/_gpu/_routines_linalg.pyx']),
    ('cupy._core._routines_logic', ['cupy/_core/_routines_logic.pyx']),
    ('cupy._core._routines_manipulation',
     ['cupy/_core/_routines_manipulation.pyx']),
    ('cupy._core._routines_math', ['cupy/_core/_gpu/_routines_math.pyx']),
    ('cupy._core._routines_sorting', ['cupy/_core/_gpu/_routines_sorting.pyx']),
    ('cupy._core._routines_statistics',
     ['cupy/_core/_routines_statistics.pyx']),
    'cupy._core.numpy_allocator',
    ('cupy._core.raw', ['cupy/_core/_gpu/raw.pyx']),
    'cupy.cuda.texture',
    'cupy.fft._cache',
    'cupy.fft._callback',
    'cupy.lib._polynomial',
    'cupy._util',
    'cupyx.scipy.ndimage._bbox_slices',
]


class CUDA_cuda(Feature):
    minimum_cuda_version = 11020

    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.name = 'cuda'
        self.required = True
        self.modules = cuda_files
        self.includes = [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'cufft.h',
            'curand.h',
            'cusparse.h',
        ]
        self.libraries = (
            # CUDA Runtime
            _cudart_static_libs +

            # CUDA Toolkit
            ['cublas', 'cufft', 'curand', 'cusparse']
        )
        self.static_libraries = ['cudart_static']
        self._version = self._UNDETERMINED

    def configure(self, compiler: Any, settings: Any) -> bool:
        try:
            out = build.build_and_run(compiler, '''
            #include <cuda.h>
            #include <stdio.h>
            int main() {
              printf("%d", CUDA_VERSION);
              return 0;
            }
            ''', include_dirs=settings['include_dirs'])  # type: ignore[no-untyped-call] # NOQA
        except Exception as e:
            utils.print_warning('Cannot check CUDA version', str(e))
            return False

        self._version = int(out)

        if self._version < self.minimum_cuda_version:
            utils.print_warning(
                'CUDA version is too old: %d' % self._version,
                'CUDA 11.2 or newer is required')
            return False
        return True


def cuda_feature_dicts(cudart_static_libs: list[str]) -> dict[str, dict[str, Any]]:
    """Return the optional CUDA library feature dicts.

    Kept as a function (rather than module-level constants) so the
    ``_cudart_static_libs`` list is evaluated once by the caller and shared.
    """
    return {
        'CUDA_cusolver': {
            'name': 'cusolver',
            'required': True,
            'file': [
                'cupy.backends.cuda.libs.cusolver',
                'cupyx.cusolver',
            ],
            'include': [
                'cusolverDn.h',
            ],
            'libraries': [
                'cusolver',
            ],
        },
        'CUDA_nccl': {
            'name': 'nccl',
            'file': [
                'cupy.backends.cuda.libs.nccl',
            ],
            'include': [
                'nccl.h',
            ],
            'libraries': [
                'nccl',
            ],
            'check_method': build.check_nccl_version,
            'version_method': build.get_nccl_version,
        },
        'CUDA_nvtx': {
            'name': 'nvtx',
            'file': [
                'cupy.backends.cuda.libs.nvtx',
            ],
            'include': [
                'nvtx3/nvToolsExt.h',
            ],
            'libraries': [
            ],
            'check_method': build.check_nvtx,
        },
        'CUDA_cutensor': {
            'name': 'cutensor',
            'file': [
                'cupy.backends.cuda.libs.cutensor',
                'cupyx.cutensor',
            ],
            'include': [
                'cutensor.h',
            ],
            'libraries': [
                'cutensor',
                'cutensorMg',
                'cublas',
            ],
            'check_method': build.check_cutensor_version,
            'version_method': build.get_cutensor_version,
        },
        'CUDA_cub': {
            'name': 'cub',
            'required': True,
            'file': [
                ('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu']),
            ],
            'include': [
                'cub/util_namespace.cuh',  # dummy
            ],
            'libraries': list(cudart_static_libs),
            'static_libraries': ['cudart_static'],
            'check_method': build.check_cub_version,
            'version_method': build.get_cub_version,
        },
        'CUDA_jitify': {
            'name': 'jitify',
            'required': True,
            'file': [
                'cupy.cuda.jitify',
            ],
            'include': [
                'cuda.h',
                'cuda_runtime.h',
                'nvrtc.h',
            ],
            'libraries': [
                # Dependency from Jitify header files
                'cuda',
                'nvrtc',
            ] + list(cudart_static_libs),
            'static_libraries': ['cudart_static'],
            'check_method': build.check_jitify_version,
            'version_method': build.get_jitify_version,
        },
        'CUDA_random': {
            'name': 'random',
            'required': True,
            'file': [
                'cupy.random._bit_generator',
                ('cupy.random._generator_api',
                 ['cupy/random/cupy_distributions.cu']),
            ],
            'include': [
            ],
            'libraries': [
                'curand',
            ] + list(cudart_static_libs),
            'static_libraries': ['cudart_static'],
        },
        'CUDA_cusparselt': {
            'name': 'cusparselt',
            'file': [
                'cupy.backends.cuda.libs.cusparselt',
            ],
            'include': [
                'cusparseLt.h',
            ],
            'libraries': [
                'cusparseLt',
            ],
            'check_method': build.check_cusparselt_version,
            'version_method': build.get_cusparselt_version,
        },
        'CUDA_thrust': {
            'name': 'thrust',
            'required': True,
            'file': [
                ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
            ],
            'include': [
                'thrust/version.h',
            ],
            'libraries': list(cudart_static_libs),
            'static_libraries': ['cudart_static'],
            'check_method': build.check_thrust_version,
            'version_method': build.get_thrust_version,
        },
        'COMMON_dlpack': {
            'name': 'dlpack',
            'required': True,
            'file': [
                'cupy._core.dlpack',
            ],
            'include': [
                'cupy/_dlpack/dlpack.h',
            ],
            'libraries': [],
        },
    }


#: Order in which the CUDA features are combined by ``get_features``.
CUDA_FEATURES_USED = [
    'CUDA_cusolver',
    'CUDA_nccl',
    'CUDA_nvtx',
    'CUDA_cutensor',
    'CUDA_cub',
    'CUDA_jitify',
    'CUDA_random',
    'CUDA_thrust',
    'CUDA_cusparselt',
    'COMMON_dlpack',
]


def get_cudart_static_libs() -> list[str]:
    """Expose the ``cudart_static`` helper libs (Linux only)."""
    return list(_cudart_static_libs)
