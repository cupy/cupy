"""AMD ROCm / HIP backend feature.

CuPy's ROCm support reuses the CUDA backend sources and links against HIP
libraries, so the feature set mirrors :mod:`cupy_builder.features.cuda` with
different library names and headers.
"""

from __future__ import annotations

from typing import Any

import cupy_builder.install_build as build

from cupy_builder.features.cuda import cuda_files


def rocm_feature_dicts() -> dict[str, dict[str, Any]]:
    """Return the ROCm/HIP feature dicts.

    The first one is named ``cuda`` (matching upstream) because CuPy's ROCm
    backend shares the ``cupy.backends.cuda`` module namespace; only the
    libraries/headers differ.
    """
    return {
        # TODO(leofang): call this "rocm" or "hip" to avoid confusion?
        'HIP_cuda_nvtx_cusolver': {
            'name': 'cuda',
            'required': True,
            'file': cuda_files + [
                'cupy.backends.cuda.libs.nvtx',
                'cupy.backends.cuda.libs.cusolver',
                'cupyx.cusolver',
            ],
            'include': [
                'hip/hip_runtime_api.h',
                'hip/hiprtc.h',
                'hipblas.h',
                'hiprand/hiprand.h',
                'hipsparse.h',
                'hipfft.h',
                'roctx.h',
                'rocsolver.h',
            ],
            'libraries': [
                'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
                'hipblas',
                ('hipfft', lambda hip_version: hip_version >= 401),
                'hiprand',
                'hipsparse',
                'rocfft',
                'roctx64',
                'rocblas',
                'rocsolver',
                'rocsparse',
            ],
            'check_method': build.check_hip_version,
            'version_method': build.get_hip_version,
        },
        'HIP_cub': {
            'name': 'cub',
            'required': True,
            'file': [
                ('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu']),
            ],
            'include': [
                'hipcub/hipcub_version.hpp',  # dummy
            ],
            'libraries': [
                'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            ],
            'check_method': build.check_cub_version,
            'version_method': build.get_cub_version,
        },
        'HIP_nccl': {
            'name': 'nccl',
            'file': [
                'cupy.backends.cuda.libs.nccl',
            ],
            'include': [
                'rccl.h',
            ],
            'libraries': [
                'rccl',
            ],
            'check_method': build.check_nccl_version,
            'version_method': build.get_nccl_version,
        },
        'HIP_random': {
            'name': 'random',
            'required': True,
            'file': [
                'cupy.random._bit_generator',
                ('cupy.random._generator_api',
                 ['cupy/random/cupy_distributions.cu']),
            ],
            'include': [
                'hiprand/hiprand.h',
            ],
            'libraries': [
                # Dependency from cuRAND header files
                'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
                'hiprand',
            ],
            'check_method': build.check_hip_version,
            'version_method': build.get_hip_version,
        },
        'HIP_thrust': {
            'name': 'thrust',
            'required': True,
            'file': [
                ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
            ],
            'include': [
                # WAR #9098:
                # rocThrust 3.3.0 (ROCm 6.4.0) cannot be compiled by host
                # compiler
                # 'thrust/version.h',
            ],
            'libraries': [
                'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            ],
        },
    }


#: Order in which the ROCm features are combined by ``get_features``.
#: ``COMMON_dlpack`` is appended by the caller.
ROCM_FEATURE_ORDER = [
    'HIP_cuda_nvtx_cusolver',
    'HIP_cub',
    'HIP_nccl',
    'HIP_random',
    'HIP_thrust',
]
