from __future__ import annotations

import os

import cupy
import pytest


_code_base = '''
#include <cuda_fp16.h>

#include <cupy/carray.cuh>
#include <cupy/complex.cuh>
#include <cupy/cuComplex_bridge.h>
#include <cupy/math_constants.h>
#include <cupy/atomics.cuh>
#include <cupy/hip_workaround.cuh>
'''

_code_nvcc = _code_base + '''
#include <cupy/type_dispatcher.cuh>

int main() {
    return 0;
}
'''

_code_nvrtc = _code_base + '''

__device__ void kernel() {
}
'''


@pytest.mark.skipif(not cupy.cuda.runtime.is_hip, reason='for HIP/ROCm')
class TestHIPWarpIntrinsics:
    """Regression test for warp intrinsics via HIPRTC on ROCm 6.2+.

    HIPRTC does not include the HIP runtime headers that provide native
    __shfl_*_sync variants. CuPy's hip_workaround.cuh must define macros
    that map these to the mask-less __shfl_* equivalents.
    See: https://github.com/cupy/cupy/issues/9829
    """

    def test_warp_intrinsics_compile_and_run(self):
        warp_size = cupy.cuda.runtime.getDeviceProperties(0)['warpSize']
        header_dir = cupy._core.core._get_header_dir_path()
        # Kernel that exercises all 5 warp intrinsics from hip_workaround.cuh.
        # Each thread in lane 0 writes results to output to verify correctness.
        code = f'''
#include <cupy/hip_workaround.cuh>

extern "C" __global__
void test_warp_intrinsics(int* out) {{
    int lane = threadIdx.x % {warp_size};
    int val = lane + 1;

    // __shfl_sync: broadcast lane 0's value to all lanes
    int r0 = __shfl_sync(0xffffffff, val, 0, {warp_size});

    // __shfl_up_sync: shift values up by 1 lane
    int r1 = __shfl_up_sync(0xffffffff, val, 1, {warp_size});

    // __shfl_down_sync: shift values down by 1 lane
    int r2 = __shfl_down_sync(0xffffffff, val, 1, {warp_size});

    // __shfl_xor_sync: XOR shuffle with mask 1 (swap neighbors)
    int r3 = __shfl_xor_sync(0xffffffff, val, 1, {warp_size});

    // __syncwarp
    __syncwarp();

    if (lane == 0) {{
        out[0] = r0;  // shfl from lane 0: should be 1
        out[1] = r1;  // shfl_up lane 0: no source, stays as val=1
        out[2] = r2;  // shfl_down lane 0: gets lane 1's value=2
        out[3] = r3;  // shfl_xor lane 0 ^ 1 = lane 1: value=2
    }}
}}
'''
        kern = cupy.RawKernel(
            code, 'test_warp_intrinsics',
            options=(f'-I{header_dir}',))
        out = cupy.zeros(4, dtype=cupy.int32)
        kern((1,), (warp_size,), (out,))
        result = out.get()
        assert result[0] == 1, f'__shfl_sync: expected 1, got {result[0]}'
        assert result[1] == 1, f'__shfl_up_sync: expected 1, got {result[1]}'
        assert result[2] == 2, f'__shfl_down_sync: expected 2, got {result[2]}'
        assert result[3] == 2, f'__shfl_xor_sync: expected 2, got {result[3]}'


@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason='for CUDA')
class TestIncludesCompileCUDA:
    def _get_cuda_archs(self):
        cuda_ver = cupy.cuda.runtime.runtimeGetVersion()
        to_exclude = set(int(a) for a in cupy.cuda.compiler._tegra_archs)
        if cuda_ver < 11000:
            # CUDA 10.2 (Tegra excluded)
            archs = (30, 35, 50, 52, 60, 61, 70, 75)
        elif cuda_ver < 11010:
            # CUDA 11.0
            archs = (35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80)
        elif cuda_ver < 11020:
            # CUDA 11.1
            archs = (35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86)
        else:
            # CUDA 11.2+
            archs = cupy.cuda.nvrtc.getSupportedArchs()
            if cuda_ver == 11020 or cuda_ver >= 12000:
                to_exclude.add(69)
        archs = tuple(set(archs) - to_exclude)

        return archs

    def _get_options(self):
        return (
            '-std=c++17',
            *cupy._core.core._get_cccl_include_options(),
            '-I{}'.format(cupy._core.core._get_header_dir_path()),
            '-I{}'.format(os.path.join(cupy.cuda.get_cuda_path(), 'include')),
        )

    def test_nvcc(self):
        options = self._get_options()
        for arch in self._get_cuda_archs():
            cupy.cuda.compiler.compile_using_nvcc(
                _code_nvcc, options=options, arch=arch)

    def test_nvrtc(self, monkeypatch):
        cuda_ver = cupy.cuda.runtime.runtimeGetVersion()
        options = self._get_options()
        for arch in self._get_cuda_archs():
            monkeypatch.setattr(
                cupy.cuda.compiler,
                "_get_arch_for_options_for_nvrtc",
                lambda _: (f'-arch=compute_{arch}', 'ptx'),
            )
            cupy.cuda.compiler.compile_using_nvrtc(
                _code_nvrtc, options=options)

            if cuda_ver >= 11010:
                monkeypatch.setattr(
                    cupy.cuda.compiler,
                    "_get_arch_for_options_for_nvrtc",
                    lambda _: (f'-arch=sm_{arch}', 'cubin'),
                )
                cupy.cuda.compiler.compile_using_nvrtc(
                    _code_nvrtc, options=options)
