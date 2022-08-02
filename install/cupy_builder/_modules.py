from typing import Any, Dict, List

import cupy_builder.install_build as build
from cupy_builder import Context

# The value of the key 'file' is a list that contains extension names
# or tuples of an extension name and a list of other souces files
# required to build the extension such as .cpp files and .cu files.
#
#   <extension name> | (<extension name>, a list of <other source>)
#
# The extension name is also interpreted as the name of the Cython
# source file required to build the extension with appending '.pyx'
# file extension.

_cuda_files = [
    'cupy_backends.cuda.api.driver',
    'cupy_backends.cuda.api._driver_enum',
    'cupy_backends.cuda.api.runtime',
    'cupy_backends.cuda.api._runtime_enum',
    'cupy_backends.cuda.libs.cublas',
    'cupy_backends.cuda.libs.curand',
    'cupy_backends.cuda.libs.cusparse',
    'cupy_backends.cuda.libs.nvrtc',
    'cupy_backends.cuda.libs.profiler',
    'cupy_backends.cuda.stream',
    'cupy_backends.cuda._softlink',
    'cupy._core._accelerator',
    'cupy._core._carray',
    'cupy._core._cub_reduction',
    'cupy._core._dtype',
    'cupy._core._fusion_kernel',
    'cupy._core._fusion_thread_local',
    'cupy._core._fusion_trace',
    'cupy._core._fusion_variable',
    'cupy._core._kernel',
    'cupy._core._memory_range',
    'cupy._core._optimize_config',
    'cupy._core._reduction',
    'cupy._core._routines_binary',
    'cupy._core._routines_indexing',
    'cupy._core._routines_linalg',
    'cupy._core._routines_logic',
    'cupy._core._routines_manipulation',
    'cupy._core._routines_math',
    'cupy._core._routines_sorting',
    'cupy._core._routines_statistics',
    'cupy._core._scalar',
    'cupy._core.core',
    'cupy._core.flags',
    'cupy._core.internal',
    'cupy._core.fusion',
    'cupy._core.new_fusion',
    'cupy._core.raw',
    'cupy.cuda.common',
    'cupy.cuda.cufft',
    'cupy.cuda.device',
    'cupy.cuda.memory',
    'cupy.cuda.memory_hook',
    'cupy.cuda.pinned_memory',
    'cupy.cuda.function',
    'cupy.cuda.stream',
    'cupy.cuda.graph',
    'cupy.cuda.texture',
    'cupy.fft._cache',
    'cupy.fft._callback',
    'cupy.lib._polynomial',
    'cupy._util',
]


def get_modules(context: Context) -> List[Dict[str, Any]]:
    # We handle nvtx (and likely any other future support) here, because
    # the HIP stubs (hip/cupy_*.h) would cause many symbols
    # to leak into all these modules even if unused. It's easier for all of
    # them to link to the same set of shared libraries.
    HIP_cuda_nvtx_cusolver = {
        # TODO(leofang): call this "rocm" or "hip" to avoid confusion?
        'name': 'cuda',
        'required': True,
        'file': _cuda_files + [
            'cupy_backends.cuda.libs.nvtx',
            'cupy_backends.cuda.libs.cusolver',
            'cupy.cusolver',
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
    }
    CUDA_cuda = {
        'name': 'cuda',
        'required': True,
        'file': _cuda_files,
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'cufft.h',
            'curand.h',
            'cusparse.h',
            'nvrtc.h',
        ],
        # TODO(kmaehashi): Split profiler module to remove dependency to
        # cudart when using CUDA Python.
        'libraries':
            (['cudart'] if context.use_cuda_python else ['cuda', 'cudart']) + [
                'cublas',
                'cufft',
                'curand',
                'cusparse',
                'nvrtc',
        ],
        'check_method': build.check_cuda_version,
        'version_method': build.get_cuda_version,
    }
    CUDA_cusolver = {
        'name': 'cusolver',
        'required': True,
        'file': [
            'cupy_backends.cuda.libs.cusolver',
            'cupy.cusolver',
        ],
        'include': [
            'cusolverDn.h',
        ],
        'libraries': [
            'cusolver',
        ],
        'check_method': build.check_cuda_version,
    }
    CUDA_cudnn = {
        'name': 'cudnn',
        'file': [
            'cupy_backends.cuda.libs.cudnn',
            'cupy.cudnn',
        ],
        'include': [
            'cudnn.h',
        ],
        'libraries': [
            'cudnn',
        ],
        'check_method': build.check_cudnn_version,
        'version_method': build.get_cudnn_version,
    }
    CUDA_nccl = {
        'name': 'nccl',
        'file': [
            'cupy_backends.cuda.libs.nccl',
        ],
        'include': [
            'nccl.h',
        ],
        'libraries': [
            'nccl',
        ],
        'check_method': build.check_nccl_version,
        'version_method': build.get_nccl_version,
    }
    CUDA_nvtx = {
        'name': 'nvtx',
        'file': [
            'cupy_backends.cuda.libs.nvtx',
        ],
        'include': [
            'nvToolsExt.h',
        ],
        'libraries': [
            'nvToolsExt' if not build.PLATFORM_WIN32 else 'nvToolsExt64_1',
        ],
        'check_method': build.check_nvtx,
    }
    CUDA_cutensor = {
        'name': 'cutensor',
        'file': [
            'cupy_backends.cuda.libs.cutensor',
            'cupy.cutensor',
        ],
        'include': [
            'cutensor.h',
        ],
        'libraries': [
            'cutensor',
            'cublas',
        ],
        'check_method': build.check_cutensor_version,
        'version_method': build.get_cutensor_version,
    }
    CUDA_cub = {
        'name': 'cub',
        'required': True,
        'file': [
            ('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu']),
        ],
        'include': [
            'cub/util_namespace.cuh',  # dummy
        ],
        'libraries': [
            # Dependency from CUB header files
            'cudart',
        ],
        'check_method': build.check_cub_version,
        'version_method': build.get_cub_version,
    }
    CUDA_jitify = {
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
            'cudart',
            'nvrtc',
        ],
        'check_method': build.check_jitify_version,
        'version_method': build.get_jitify_version,
    }
    CUDA_random = {
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
            # Dependency from cuRAND header files
            'cudart',
            'curand',
        ],
    }
    HIP_random = {
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
    }
    CUDA_cusparselt = {
        'name': 'cusparselt',
        'file': [
            'cupy_backends.cuda.libs.cusparselt',
        ],
        'include': [
            'cusparseLt.h',
        ],
        'libraries': [
            'cusparseLt',
        ],
        'check_method': build.check_cusparselt_version,
        'version_method': build.get_cusparselt_version,
    }
    HIP_cub = {
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
    }
    HIP_nccl = {
        'name': 'nccl',
        'file': [
            'cupy_backends.cuda.libs.nccl',
        ],
        'include': [
            'rccl.h',
        ],
        'libraries': [
            'rccl',
        ],
        'check_method': build.check_nccl_version,
        'version_method': build.get_nccl_version,
    }
    HIP_thrust = {
        'name': 'thrust',
        'required': True,
        'file': [
            ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
        ],
        'include': [
            'thrust/version.h',
        ],
        'libraries': [
            'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
        ],
    }
    CUDA_thrust = {
        'name': 'thrust',
        'required': True,
        'file': [
            ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
        ],
        'include': [
            'thrust/device_ptr.h',
            'thrust/sequence.h',
            'thrust/sort.h',
        ],
        'libraries': [
            # Dependency from Thrust header files
            'cudart',
        ],
        'check_method': build.check_thrust_version,
        'version_method': build.get_thrust_version,
    }
    COMMON_dlpack = {
        'name': 'dlpack',
        'required': True,
        'file': [
            'cupy._core.dlpack',
        ],
        'include': [
            'cupy/dlpack/dlpack.h',
        ],
        'libraries': [],
    }

    if context.use_hip:
        return [
            HIP_cuda_nvtx_cusolver,
            HIP_cub,
            HIP_nccl,
            HIP_random,
            HIP_thrust,
            COMMON_dlpack,
        ]

    return [
        CUDA_cuda,
        CUDA_cusolver,
        CUDA_cudnn,
        CUDA_nccl,
        CUDA_nvtx,
        CUDA_cutensor,
        CUDA_cub,
        CUDA_jitify,
        CUDA_random,
        CUDA_thrust,
        CUDA_cusparselt,
        COMMON_dlpack,
    ]
