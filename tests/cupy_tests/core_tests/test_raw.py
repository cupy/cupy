from __future__ import annotations

import contextlib
import io
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
from unittest import mock

import pytest

import cupy
from cupy import testing
from cupy import _util
from cupy._core import _accelerator
from cupy.cuda import compiler
from cupy.cuda import _compiler_cache
from cupy.cuda import memory
from cupy_backends.cuda.libs import nvrtc


_test_source1 = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        y[tid] = x1[tid] + x2[tid];
}
'''

_test_compile_src = r'''
extern "C" __global__
void test_op(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int j;  // To generate a warning to appear in the log stream
    if (tid < N)
        y[tid] = x1[tid] OP x2[tid];
}
'''

# test compiling and invoking multiple kernels in one single .cubin
_test_source2 = r'''
extern "C"{

__global__ void test_sum(const float* x1, const float* x2, float* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}
'''

# test C macros
_test_source3 = r'''
#ifndef PRECISION
    #define PRECISION 2
#endif

#if PRECISION == 2
    #define TYPE double
#elif PRECISION == 1
    #define TYPE float
#else
    #error precision not supported
#endif

extern "C"{

__global__ void test_sum(const TYPE* x1, const TYPE* x2, TYPE* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const TYPE* x1, const TYPE* x2, TYPE* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}
'''

# dynamic parallelism
_test_source4 = r'''
extern "C"{

__global__ void test_kernel_inner(float *arr, int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        arr[tid] = 1.0;
}

__global__ void test_kernel(float *arr, int N, int inner_blk)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N/inner_blk)
        test_kernel_inner<<<1, inner_blk>>>(arr+tid*inner_blk, inner_blk);
}

}
'''

# to generate cubin/ptx
_test_source5 = r'''
extern "C" __global__
void test_div(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        y[tid] = x1[tid] / (x2[tid] + 1.0);
}
'''

_test_cuComplex = r'''
#include <cuComplex.h>
#define N 100

extern "C"{
/* ------------------- double complex ------------------- */

__global__ void test_add(cuDoubleComplex* arr1, cuDoubleComplex* arr2,
                         cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCadd(arr1[tid], arr2[tid]);
    }
}

__global__ void test_sub(cuDoubleComplex* arr1, cuDoubleComplex* arr2,
                         cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCsub(arr1[tid], arr2[tid]);
    }
}

__global__ void test_mul(cuDoubleComplex* arr1, cuDoubleComplex* arr2,
                         cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCmul(arr1[tid], arr2[tid]);
    }
}

__global__ void test_div(cuDoubleComplex* arr1, cuDoubleComplex* arr2,
                         cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCdiv(arr1[tid], arr2[tid]);
    }
}

__global__ void test_conj(cuDoubleComplex* arr, cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuConj(arr[tid]);
    }
}

__global__ void test_abs(cuDoubleComplex* arr, double* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCabs(arr[tid]);
    }
}

__global__ void test_fma(cuDoubleComplex* A, cuDoubleComplex* B,
                         cuDoubleComplex* C, cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCfma(A[tid], B[tid], C[tid]);
    }
}

__global__ void test_make(cuDoubleComplex* arr) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    cuDoubleComplex out = make_cuDoubleComplex(1.8, 2.9);
    if (tid < N) {
        arr[tid] = make_cuDoubleComplex(cuCreal(out), -3.*cuCimag(out));
    }
}

__global__ void test_downcast(cuDoubleComplex* arr, cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuComplexDoubleToFloat(arr[tid]);
    }
}

__global__ void test_add_scalar(cuDoubleComplex* arr, cuDoubleComplex scalar,
                                cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCadd(arr[tid], scalar);
    }
}

/* ------------------- single complex ------------------- */

__global__ void test_addf(cuComplex* arr1, cuComplex* arr2,
                          cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCaddf(arr1[tid], arr2[tid]);
    }
}

__global__ void test_subf(cuComplex* arr1, cuComplex* arr2,
                          cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCsubf(arr1[tid], arr2[tid]);
    }
}

__global__ void test_mulf(cuComplex* arr1, cuComplex* arr2,
                          cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCmulf(arr1[tid], arr2[tid]);
    }
}

__global__ void test_divf(cuComplex* arr1, cuComplex* arr2,
                          cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCdivf(arr1[tid], arr2[tid]);
    }
}

__global__ void test_conjf(cuComplex* arr, cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuConjf(arr[tid]);
    }
}

__global__ void test_absf(cuFloatComplex* arr, float* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCabsf(arr[tid]);
    }
}

__global__ void test_fmaf(cuFloatComplex* A, cuFloatComplex* B,
                          cuFloatComplex* C, cuFloatComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCfmaf(A[tid], B[tid], C[tid]);
    }
}

__global__ void test_makef(cuComplex* arr) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    cuComplex out = make_cuFloatComplex(1.8, 2.9);
    if (tid < N) {
        arr[tid] = make_cuFloatComplex(cuCrealf(out), -3.*cuCimagf(out));
    }
}

__global__ void test_upcast(cuComplex* arr, cuDoubleComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuComplexFloatToDouble(arr[tid]);
    }
}

__global__ void test_addf_scalar(cuComplex* arr, cuComplex scalar,
                                 cuComplex* out) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = cuCadd(arr[tid], scalar);
    }
}

}
'''

test_const_mem = r'''
extern "C"{
__constant__ float some_array[100];

__global__ void multiply_by_const(float* x, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N) {
        x[id] *= some_array[id];
    }
}
}
'''

test_cxx_template = r'''
#include <cupy/complex.cuh>

template<typename T>
__global__ void my_sqrt(T* input, int N) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    input[x] *= input[x];
  }
}

__global__ void my_func(double* input, int N) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    input[x] *= input[x];
  }
}
'''

test_cast = r'''
extern "C" __global__ void my_func(void* input, int N) {
  double* arr = (double*)(input);
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    arr[x] = 3.0 * arr[x] - 8.0;
  }
}
'''


@contextlib.contextmanager
def use_temporary_cache_dir():
    # Note uses mock, so not thread-safe (except at class/method level)
    # tempdir fixture could be used instead.
    target1 = 'cupy.cuda.compiler._kernel_cache_backend'
    target2 = 'cupy.cuda.compiler._empty_file_preprocess_cache'
    temp_cache = {}
    with tempfile.TemporaryDirectory() as path:
        with mock.patch(target1, _compiler_cache.DiskKernelCacheBackend(path)):
            with mock.patch(target2, temp_cache):
                yield path


@contextlib.contextmanager
def compile_in_memory(in_memory):
    target = 'cupy.cuda.compiler._get_bool_env_variable'

    def new_target(name, default):
        if name == 'CUPY_CACHE_IN_MEMORY':
            return in_memory
        else:
            # below is the source code of _get_bool_env_variable
            val = os.environ.get(name)
            if val is None or len(val) == 0:
                return default
            try:
                return int(val) == 1
            except ValueError:
                return False

    with mock.patch(target, new_target) as m:
        yield m


def find_nvcc_ver():
    nvcc_ver_pattern = r'release (\d+\.\d+)'
    cmd = cupy.cuda.get_nvcc_path().split()
    cmd += ['--version']

    output = compiler._run_cc(cmd, None, 'nvcc')
    match = re.search(nvcc_ver_pattern, output)
    assert match

    # convert to driver ver format
    major, minor = match.group(1).split('.')
    return int(major) * 1000 + int(minor) * 10


# Recent CCCL has made Jitify cold-launch very slow, see the discussion
# starting https://github.com/cupy/cupy/pull/8899#issuecomment-2613022424.
no_jitify_markers = [
    pytest.mark.filterwarnings(
        "ignore:The jitify argument is deprecated:DeprecationWarning")
]
jitify_markers = [
    pytest.mark.filterwarnings(
        "ignore:jitify=True is deprecated:DeprecationWarning"),
    pytest.mark.thread_unsafe(
        reason="Jitify seems to have problems, skip as largely unmaintained.")
]


@pytest.mark.parametrize("backend,in_memory,clean_up,jitify", [
    # First test NVRTC
    pytest.param(
        "nvrtc", False, False, False,
        marks=no_jitify_markers,
        id="NVRTC (no-jitify)"),
    # this run will read from in-memory cache
    pytest.param(
        "nvrtc", True, False, False,
        marks=no_jitify_markers,
        id="NVRTC (in-memory cache, no-jitify)"),
    # this run will force recompilation
    pytest.param(
        "nvrtc", True, True, False,
        marks=no_jitify_markers,
        id="NVRTC (in-memory cache, re-compile, no-jitify)"),
    # Finally, we test NVCC
    pytest.param(
        "nvcc", False, False, False,
        marks=no_jitify_markers,
        id="NVCC (no-jitify)"),

    # Below is the same set of NVRTC tests, with Jitify turned on.
    # For tests that can already pass, it shouldn't matter whether
    # Jitify is on or not, and the only side effect is to add overhead.
    # It doesn't make sense to test NVCC + Jitify.
    pytest.param(
        "nvrtc", False, False, True,
        marks=jitify_markers,
        id="NVRTC (jitify)"),
    pytest.param(
        "nvrtc", True, False, True,
        marks=jitify_markers,
        id="NVRTC (in-memory cache, jitify)"),
    pytest.param(
        "nvrtc", True, True, True,
        marks=jitify_markers,
        id="NVRTC (in-memory cache, re-compile, jitify)"),
])
class TestRaw:

    _nvcc_ver = None
    _nvrtc_ver = None

    @pytest.fixture(autouse=True)
    def configure(self, backend, in_memory, clean_up, jitify):
        if clean_up:
            if cupy.cuda.runtime.is_hip:
                # Clearing memo triggers recompiling kernels using name
                # expressions in other tests, e.g. dot and matmul, which
                # hits a nvrtc bug. See #5843, #5945 and #6725.
                pytest.skip('Clearing memo hits a nvrtc bug in other tests')
            _util.clear_memo()
        self.dev = cupy.cuda.runtime.getDevice()
        assert self.dev != 1

        if cupy.cuda.runtime.is_hip and jitify:
            pytest.skip('Jitify does not support ROCm/HIP')

        # TODO: how is this used? could benefit from context manager
        self.in_memory_context = compile_in_memory(in_memory)
        self.in_memory_context.__enter__()

        kern = cupy.RawKernel(
            _test_source1, 'test_sum',
            backend=backend, jitify=jitify)
        mod2 = cupy.RawModule(
            code=_test_source2,
            backend=backend, jitify=jitify)
        mod3 = cupy.RawModule(
            code=_test_source3,
            options=('-DPRECISION=2',),
            backend=backend, jitify=jitify)

        with use_temporary_cache_dir() as cache_dir:
            yield kern, mod2, mod3, cache_dir

            if (in_memory
                    and _accelerator.ACCELERATOR_CUB not in
                    _accelerator.get_reduction_accelerators()):
                # should not write any file to the cache dir,
                # but the CUB reduction
                # kernel uses nvcc, with which I/O cannot be avoided
                files = os.listdir(cache_dir)
                for f in files:
                    # only test_load_cubin_*.cu files should be present
                    assert re.match(r'test_load_cubin_(\d+)\.cu', f)

        self.in_memory_context.__exit__(*sys.exc_info())

    def _helper(self, kernel, dtype):
        N = 10
        x1 = cupy.arange(N**2, dtype=dtype).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=dtype)
        y = cupy.zeros((N, N), dtype=dtype)
        kernel((N,), (N,), (x1, x2, y, N**2))
        return x1, x2, y

    def test_basic(self, configure):
        kern, _, _, _ = configure
        x1, x2, y = self._helper(kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

    def test_kernel_attributes(self, configure):
        kern, _, _, _ = configure
        attrs = kern.attributes
        for attribute in ['binary_version',
                          'cache_mode_ca',
                          'const_size_bytes',
                          'local_size_bytes',
                          'max_dynamic_shared_size_bytes',
                          'max_threads_per_block',
                          'num_regs',
                          'preferred_shared_memory_carveout',
                          'ptx_version',
                          'shared_size_bytes']:
            assert attribute in attrs
        # TODO(leofang): investigate why this fails on ROCm 3.5.0
        if not cupy.cuda.runtime.is_hip:
            assert kern.num_regs > 0
        assert kern.max_threads_per_block > 0
        assert kern.shared_size_bytes == 0

    def test_module(self, configure):
        _, mod2, _, _ = configure
        module = mod2
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        x1, x2, y = self._helper(ker_sum, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

        x1, x2, y = self._helper(ker_times, cupy.float32)
        assert cupy.allclose(y, x1 * x2)

    def test_compiler_flag(self, configure):
        _, _, mod3, _ = configure
        module = mod3
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        x1, x2, y = self._helper(ker_sum, cupy.float64)
        assert cupy.allclose(y, x1 + x2)

        x1, x2, y = self._helper(ker_times, cupy.float64)
        assert cupy.allclose(y, x1 * x2)

    def test_invalid_compiler_flag(self, backend, jitify):
        if cupy.cuda.runtime.is_hip and backend == 'nvrtc':
            pytest.skip('hiprtc does not handle #error macro properly')

        if jitify:
            ex_type = cupy.cuda.compiler.JitifyException
        else:
            ex_type = cupy.cuda.compiler.CompileException

        with pytest.raises(ex_type) as ex:
            mod = cupy.RawModule(code=_test_source3,
                                 options=('-DPRECISION=3',),
                                 backend=backend,
                                 jitify=jitify)
            mod.get_function('test_sum')  # enforce compilation

        if not jitify:
            assert 'precision not supported' in str(ex.value)

    def _find_nvcc_ver(self):
        if self._nvcc_ver:
            return self._nvcc_ver

        self._nvcc_ver = find_nvcc_ver()
        return self._nvcc_ver

    def _find_nvrtc_ver(self):
        if self._nvrtc_ver:
            return self._nvrtc_ver

        # convert to driver ver format
        major, minor = nvrtc.getVersion()
        self._nvrtc_ver = int(major) * 1000 + int(minor) * 10
        return self._nvrtc_ver

    def _check_ptx_loadable(self, compiler: str):
        # if the PTX version is higher than the driver version, it won't
        # be either JIT'able (CUDA_ERROR_UNSUPPORTED_PTX_VERSION) or loadable
        # (CUDA_ERROR_NO_BINARY_FOR_GPU), see the table at
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes-ptx-release-history
        if compiler == "nvrtc":
            compiler_ver = self._find_nvrtc_ver()
        elif compiler == "nvcc":
            compiler_ver = self._find_nvcc_ver()
        driver_ver = cupy.cuda.runtime.driverGetVersion()
        if driver_ver < compiler_ver:
            raise pytest.skip()

    def _generate_file(self, cache_dir, ext: str):
        # generate cubin/ptx by calling nvcc/hipcc

        if not cupy.cuda.runtime.is_hip:
            cc = cupy.cuda.get_nvcc_path()
            arch = '-gencode=arch=compute_{CC},code=sm_{CC}'.format(
                CC=compiler._get_arch())
            code = _test_source5
        else:
            # TODO(leofang): expose get_hipcc_path() to cupy.cuda?
            cc = cupy._environment.get_hipcc_path()
            arch = '-v'  # dummy
            code = compiler._convert_to_hip_source(_test_source5, None, False)
        # split() is needed because nvcc could come from the env var NVCC
        cmd = cc.split()
        thread_id = threading.get_ident()
        source = f'{cache_dir}/test_load_cubin_{thread_id}.cu'
        file_path = cache_dir + f'test_load_cubin_{thread_id}'
        with open(source, 'w') as f:
            f.write(code)
        if not cupy.cuda.runtime.is_hip:
            if ext == 'cubin':
                file_path += '.cubin'
                flag = '-cubin'
            elif ext == 'ptx':
                file_path += '.ptx'
                flag = '-ptx'
            else:
                raise ValueError
        else:
            file_path += '.hsaco'
            flag = '--genco'
        cmd += [arch, flag, source, '-o', file_path]
        cc = 'nvcc' if not cupy.cuda.runtime.is_hip else 'hipcc'
        compiler._run_cc(cmd, cache_dir, cc)

        return file_path

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="HIP uses hsaco, not cubin")
    def test_load_cubin(self, configure, backend):
        _, _, _, cache_dir = configure
        # generate cubin in the temp dir
        file_path = self._generate_file(cache_dir, 'cubin')

        # load cubin and test the kernel
        mod = cupy.RawModule(path=file_path, backend=backend)
        ker = mod.get_function('test_div')
        x1, x2, y = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="HIP uses hsaco, not ptx")
    def test_load_ptx(self, configure, backend):
        _, _, _, cache_dir = configure
        # use nvcc to generate ptx in the temp dir
        self._check_ptx_loadable('nvcc')
        file_path = self._generate_file(cache_dir, 'ptx')

        # load ptx and test the kernel
        mod = cupy.RawModule(path=file_path, backend=backend)
        ker = mod.get_function('test_div')
        x1, x2, y = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    @pytest.mark.skipif(
        not cupy.cuda.runtime.is_hip,
        reason="CUDA uses cubin/ptx, not hsaco")
    def test_load_hsaco(self, configure, backend):
        _, _, _, cache_dir = configure
        # generate hsaco in the temp dir
        file_path = self._generate_file(cache_dir, 'hsaco')

        # load cubin and test the kernel
        mod = cupy.RawModule(path=file_path, backend=backend)
        ker = mod.get_function('test_div')
        x1, x2, y = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    def test_module_load_failure(self, backend):
        # in principle this test is better done in test_driver.py, but
        # this error is more likely to appear when using RawModule, so
        # let us do it here
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            mod = cupy.RawModule(
                path=os.path.expanduser('~/this_does_not_exist.cubin'),
                backend=backend)
            mod.get_function('nonexisting_kernel')  # enforce loading
        assert ('CUDA_ERROR_FILE_NOT_FOUND' in str(ex.value)  # CUDA
                or 'hipErrorFileNotFound' in str(ex.value))  # HIP

    def test_module_neither_code_nor_path(self):
        with pytest.raises(TypeError):
            cupy.RawModule()

    def test_module_both_code_and_path(self):
        with pytest.raises(TypeError):
            cupy.RawModule(
                code=_test_source1,
                path='test.cubin')

    def test_get_function_failure(self, configure):
        _, mod2, _, _ = configure
        # in principle this test is better done in test_driver.py, but
        # this error is more likely to appear when using RawModule, so
        # let us do it here
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            mod2.get_function('no_such_kernel')
        assert ('CUDA_ERROR_NOT_FOUND' in str(ex.value)  # for CUDA
                or 'hipErrorNotFound' in str(ex.value))  # for HIP

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="ROCm/HIP does not support dynamic parallelism")
    def test_dynamical_parallelism(self, backend, jitify):
        self._check_ptx_loadable('nvrtc')
        ker = cupy.RawKernel(_test_source4, 'test_kernel', options=('-dc',),
                             backend=backend, jitify=jitify)
        N = 169
        inner_chunk = 13
        x = cupy.zeros((N,), dtype=cupy.float32)
        ker((1,), (N//inner_chunk,), (x, N, inner_chunk))
        assert (x == 1.0).all()

    def test_dynamical_parallelism_compile_failure(self, backend, jitify):
        # no option for separate compilation is given should cause an error
        ker = cupy.RawKernel(_test_source4, 'test_kernel',
                             backend=backend, jitify=jitify)
        N = 10
        inner_chunk = 2
        x = cupy.zeros((N,), dtype=cupy.float32)
        use_ptx = os.environ.get(
            'CUPY_COMPILE_WITH_PTX', False)
        if jitify:
            error = cupy.cuda.compiler.JitifyException
        elif backend == 'nvrtc' and (
                use_ptx or
                (cupy.cuda.driver._is_cuda_python()
                 and cupy.cuda.runtime.runtimeGetVersion() < 11010) or
                (not cupy.cuda.driver._is_cuda_python()
                 and not cupy.cuda.runtime.is_hip
                 and cupy.cuda.driver.get_build_version() < 11010)):
            # raised when calling ls.complete()
            error = cupy.cuda.driver.CUDADriverError
        else:  # nvcc, hipcc, hiprtc
            error = cupy.cuda.compiler.CompileException
        with pytest.raises(error):
            ker((1,), (N//inner_chunk,), (x, N, inner_chunk))

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason='HIP code should not use cuFloatComplex')
    def test_cuFloatComplex(self, jitify):
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex64

        mod = cupy.RawModule(
            code=_test_cuComplex,
            translate_cucomplex=True,
            jitify=jitify)
        a = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        a = a.astype(dtype)
        b = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        b = b.astype(dtype)
        c = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        c = c.astype(dtype)
        out = cupy.zeros((N,), dtype=dtype)
        out_float = cupy.zeros((N,), dtype=cupy.float32)
        out_up = cupy.zeros((N,), dtype=cupy.complex128)

        ker = mod.get_function('test_addf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

        ker = mod.get_function('test_subf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a - b).all()

        ker = mod.get_function('test_mulf')
        ker((grid,), (block,), (a, b, out))
        assert cupy.allclose(out, a * b)

        ker = mod.get_function('test_divf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a / b).all()

        ker = mod.get_function('test_conjf')
        ker((grid,), (block,), (a, out))
        assert (out == cupy.conj(a)).all()

        ker = mod.get_function('test_absf')
        ker((grid,), (block,), (a, out_float))
        assert (out_float == cupy.abs(a)).all()

        ker = mod.get_function('test_fmaf')
        ker((grid,), (block,), (a, b, c, out))
        assert cupy.allclose(out, a * b + c)

        ker = mod.get_function('test_makef')
        ker((grid,), (block,), (out,))
        # because of precision issue, the (A==B).all() semantics would fail
        assert cupy.allclose(out, 1.8 - 1j * 8.7)

        ker = mod.get_function('test_upcast')
        ker((grid,), (block,), (a, out_up))
        assert (out_up == a.astype(cupy.complex128)).all()

        # NumPy scalars.
        b = cupy.complex64(2 + 3j)
        ker = mod.get_function('test_addf_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="HIP code should not use cuDoubleComplex")
    def test_cuDoubleComplex(self, jitify):
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex128

        mod = cupy.RawModule(
            code=_test_cuComplex,
            translate_cucomplex=True,
            jitify=jitify)
        a = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        a = a.astype(dtype)
        b = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        b = b.astype(dtype)
        c = cupy.random.random((N,)) + 1j*cupy.random.random((N,))
        c = c.astype(dtype)
        out = cupy.zeros((N,), dtype=dtype)
        out_float = cupy.zeros((N,), dtype=cupy.float64)
        out_down = cupy.zeros((N,), dtype=cupy.complex64)

        ker = mod.get_function('test_add')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

        ker = mod.get_function('test_sub')
        ker((grid,), (block,), (a, b, out))
        assert (out == a - b).all()

        ker = mod.get_function('test_mul')
        ker((grid,), (block,), (a, b, out))
        assert cupy.allclose(out, a * b)

        ker = mod.get_function('test_div')
        ker((grid,), (block,), (a, b, out))
        assert (out == a / b).all()

        ker = mod.get_function('test_conj')
        ker((grid,), (block,), (a, out))
        assert (out == cupy.conj(a)).all()

        ker = mod.get_function('test_abs')
        ker((grid,), (block,), (a, out_float))
        assert (out_float == cupy.abs(a)).all()

        ker = mod.get_function('test_fma')
        ker((grid,), (block,), (a, b, c, out))
        assert cupy.allclose(out, a * b + c)

        ker = mod.get_function('test_make')
        ker((grid,), (block,), (out,))
        assert (out == 1.8 - 1j * 8.7).all()

        ker = mod.get_function('test_downcast')
        ker((grid,), (block,), (a, out_down))
        assert (out_down == a.astype(cupy.complex64)).all()

        # NumPy scalars.
        b = cupy.complex128(2 + 3j)
        ker = mod.get_function('test_add_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

        # Python scalars.
        b = 2 + 3j
        ker = mod.get_function('test_add_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

    @pytest.mark.thread_unsafe(reason="mutates global in RawModule")
    def test_const_memory(self, backend, jitify):
        mod = cupy.RawModule(code=test_const_mem,
                             backend=backend,
                             jitify=jitify)
        ker = mod.get_function('multiply_by_const')
        mem_ptr = mod.get_global('some_array')
        const_arr = cupy.ndarray((100,), cupy.float32, mem_ptr)
        data = cupy.arange(100, dtype=cupy.float32)
        const_arr[...] = data
        output_arr = cupy.ones(100, dtype=cupy.float32)
        ker((1,), (100,), (output_arr, cupy.int32(100)))
        assert (data == output_arr).all()

    def test_template_specialization(self, backend, clean_up, jitify):
        if backend == 'nvcc':
            pytest.skip(reason="nvcc does not support template specialization")

        # TODO(leofang): investigate why hiprtc generates a wrong source code
        # when the same code is compiled and discarded. It seems hiprtc has
        # an internal cache that conflicts with the 2nd compilation attempt.
        if cupy.cuda.runtime.is_hip and clean_up:
            pytest.skip(reason="skip a potential hiprtc bug")

        # compile code
        if cupy.cuda.runtime.is_hip:
            # ROCm 5.0 returns HIP_HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID for
            # my_sqrt<complex<double>>, so we use thrust::complex<double>
            # instead.
            name_expressions = ['my_sqrt<int>', 'my_sqrt<float>',
                                'my_sqrt<thrust::complex<double>>', 'my_func']
        else:
            name_expressions = ['my_sqrt<int>', 'my_sqrt<float>',
                                'my_sqrt<complex<double>>', 'my_func']
        mod = cupy.RawModule(code=test_cxx_template,
                             name_expressions=name_expressions,
                             jitify=jitify)

        dtypes = (cupy.int32, cupy.float32, cupy.complex128, cupy.float64)
        for ker_T, dtype in zip(name_expressions, dtypes):
            # get specialized kernels
            if cupy.cuda.runtime.is_hip:
                # TODO(leofang): investigate why getLoweredName has no error
                # but returns an empty string for my_sqrt<complex<double>>
                mangled_name = mod.module.mapping.get(ker_T)
                if mangled_name == '':
                    continue
            ker = mod.get_function(ker_T)

            # prepare inputs & expected outputs
            in_arr = cupy.testing.shaped_random((10,), dtype=dtype)
            out_arr = in_arr**2

            # run
            ker((1,), (10,), (in_arr, 10))

            # check results
            assert cupy.allclose(in_arr, out_arr)

    def test_template_failure(self, backend, jitify):
        name_expressions = ['my_sqrt<int>']

        # 1. nvcc is disabled for this feature
        if backend == 'nvcc':
            with pytest.raises(ValueError) as e:
                cupy.RawModule(code=test_cxx_template, backend=backend,
                               name_expressions=name_expressions)
            assert 'nvrtc' in str(e.value)
            return  # the rest of tests do not apply to nvcc

        # 2. compile code without specializations
        mod = cupy.RawModule(code=test_cxx_template,
                             jitify=jitify)
        # ...try to get a specialized kernel
        match = ('named symbol not found' if not cupy.cuda.runtime.is_hip else
                 'hipErrorNotFound')
        with pytest.raises(cupy.cuda.driver.CUDADriverError, match=match):
            mod.get_function('my_sqrt<int>')

        # 3. try to fetch something we didn't specialize for
        mod = cupy.RawModule(code=test_cxx_template,
                             name_expressions=name_expressions,
                             jitify=jitify)
        if cupy.cuda.runtime.is_hip:
            msg = 'hipErrorNotFound'
        else:
            msg = 'named symbol not found'
        with pytest.raises(cupy.cuda.driver.CUDADriverError, match=msg):
            mod.get_function('my_sqrt<double>')

    def test_raw_pointer(self, backend, jitify):
        mod = cupy.RawModule(code=test_cast,
                             backend=backend,
                             jitify=jitify)
        ker = mod.get_function('my_func')

        a = cupy.ones((100,), dtype=cupy.float64)
        memptr = memory.alloc(100 * a.dtype.itemsize)
        memptr.copy_from(a.data, 100 * a.dtype.itemsize)  # one-initialize
        b = cupy.ndarray((100,), cupy.float64, memptr=memptr)

        ker((1,), (100,), (memptr, 100))
        a = 3. * a - 8.
        assert (a == b).all()

    @testing.multi_gpu(2)
    def test_context_switch_RawKernel(self, configure):
        kern, _, _, _ = configure
        # run test_basic() on another device

        # we need to launch it once to force compiling
        x1, x2, y = self._helper(kern, cupy.float32)

        with cupy.cuda.Device(1):
            x1, x2, y = self._helper(kern, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule1(self, configure):
        _, mod2, _, _ = configure
        # run test_module() on another device
        # in this test, re-compiling happens at 2nd get_function()
        module = mod2
        with cupy.cuda.Device(0):
            module.get_function('test_sum')

        with cupy.cuda.Device(1):
            ker_sum = module.get_function('test_sum')
            x1, x2, y = self._helper(ker_sum, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule2(self, configure):
        _, mod2, _, _ = configure
        # run test_module() on another device
        # in this test, re-compiling happens at kernel launch
        module = mod2
        with cupy.cuda.Device(0):
            ker_sum = module.get_function('test_sum')

        with cupy.cuda.Device(1):
            x1, x2, y = self._helper(ker_sum, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule3(self, configure, backend):
        _, _, _, cache_dir = configure
        # run test_load_cubin() on another device
        # generate cubin in the temp dir and load it on device 0

        device0 = cupy.cuda.Device(0)
        device1 = cupy.cuda.Device(1)
        if device0.compute_capability != device1.compute_capability:
            raise pytest.skip()

        with device0:
            file_path = self._generate_file(cache_dir, 'cubin')
            mod = cupy.RawModule(path=file_path, backend=backend)
            mod.get_function('test_div')

        # in this test, reloading happens at 2nd get_function()
        with device1:
            ker = mod.get_function('test_div')
            x1, x2, y = self._helper(ker, cupy.float32)
            assert cupy.allclose(y, x1 / (x2 + 1.0))

    @testing.multi_gpu(2)
    def test_context_switch_RawModule4(self, configure, backend):
        _, _, _, cache_dir = configure
        # run test_load_cubin() on another device
        # generate cubin in the temp dir and load it on device 0

        device0 = cupy.cuda.Device(0)
        device1 = cupy.cuda.Device(1)
        if device0.compute_capability != device1.compute_capability:
            raise pytest.skip()

        with device0:
            file_path = self._generate_file(cache_dir, 'cubin')
            mod = cupy.RawModule(path=file_path, backend=backend)
            ker = mod.get_function('test_div')

        # in this test, reloading happens at kernel launch
        with device1:
            x1, x2, y = self._helper(ker, cupy.float32)
            assert cupy.allclose(y, x1 / (x2 + 1.0))

    @testing.multi_gpu(2)
    def test_context_switch_RawModule5(self, backend, jitify):
        # run test_template_specialization() on another device
        # in this test, re-compiling happens at get_function()
        if backend == 'nvcc':
            pytest.skip(reason="nvcc does not support template specialization")

        # compile code
        name_expressions = ['my_sqrt<unsigned int>']
        name = name_expressions[0]
        with cupy.cuda.Device(0):
            mod = cupy.RawModule(code=test_cxx_template,
                                 name_expressions=name_expressions,
                                 jitify=jitify)

            # get specialized kernels
            mod.get_function(name)

        # switch device
        with cupy.cuda.Device(1):
            # get specialized kernels
            ker = mod.get_function(name)

            # prepare inputs & expected outputs
            in_arr = cupy.testing.shaped_random((10,), dtype=cupy.uint32)
            out_arr = in_arr**2

            # run
            ker((1,), (10,), (in_arr, 10))

            # check results
            assert cupy.allclose(in_arr, out_arr)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule6(self, backend, jitify):
        # run test_template_specialization() on another device
        # in this test, re-compiling happens at kernel launch
        if backend == 'nvcc':
            pytest.skip(reason="nvcc does not support template specialization")

        # compile code
        name_expressions = ['my_sqrt<unsigned int>']
        name = name_expressions[0]
        with cupy.cuda.Device(0):
            mod = cupy.RawModule(code=test_cxx_template,
                                 name_expressions=name_expressions,
                                 jitify=jitify)

            # get specialized kernels
            ker = mod.get_function(name)

        # switch device
        with cupy.cuda.Device(1):
            # prepare inputs & expected outputs
            in_arr = cupy.testing.shaped_random((10,), dtype=cupy.uint32)
            out_arr = in_arr**2

            # run
            ker((1,), (10,), (in_arr, 10))

            # check results
            assert cupy.allclose(in_arr, out_arr)

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="only CUDA raises warning")
    @pytest.mark.thread_unsafe(reason="mutates global cache directory")
    def test_compile_kernel(self, backend, jitify):
        kern = cupy.RawKernel(
            _test_compile_src, 'test_op',
            options=('-DOP=+',),
            backend=backend,
            jitify=jitify)
        log = io.StringIO()
        with use_temporary_cache_dir():
            kern.compile(log_stream=log)
        assert 'warning' in log.getvalue()
        x1, x2, y = self._helper(kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

    @pytest.mark.skipif(
        cupy.cuda.runtime.is_hip,
        reason="only CUDA raises warning")
    @pytest.mark.thread_unsafe(reason="mutates global cache directory")
    def test_compile_module(self, backend, jitify):
        module = cupy.RawModule(
            code=_test_compile_src,
            backend=backend,
            options=('-DOP=+',),
            jitify=jitify)
        log = io.StringIO()
        with use_temporary_cache_dir():
            module.compile(log_stream=log)
        assert 'warning' in log.getvalue()
        kern = module.get_function('test_op')
        x1, x2, y = self._helper(kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)


_test_grid_sync = r'''
#include <cooperative_groups.h>

extern "C" __global__
void test_grid_sync(const float* x1, const float* x2, float* y, int n) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < n; i += size) {
        y[i] = x1[i];
    }
    cg::sync(grid);
    for (int i = n - 1 - tid; i >= 0; i -= size) {
        y[i] += x2[i];
    }
}
'''


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("block", [64, 256])
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip or find_nvcc_ver() >= 12020,
    reason="fp16 header compatibility issue, see cupy#8412 (Skip on HIP)")
@pytest.mark.skipif(
    9000 > cupy.cuda.runtime.runtimeGetVersion(),
    reason="Requires CUDA 9.x or later")
@pytest.mark.skipif(
    60 > int(cupy.cuda.device.get_compute_capability()),
    reason="Requires compute capability 6.0 or later")
class TestRawGridSync:

    @pytest.mark.thread_unsafe(reason="mutates global cache directory")
    def test_grid_sync_rawkernel(self, n, block):
        with use_temporary_cache_dir():
            kern_grid_sync = cupy.RawKernel(
                _test_grid_sync, 'test_grid_sync', backend='nvcc',
                enable_cooperative_groups=True)
            x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
            x2 = cupy.ones((n, n), dtype='float32')
            y = cupy.zeros((n, n), dtype='float32')
            grid = (n * n + block - 1) // block
            kern_grid_sync((grid,), (block,), (x1, x2, y, n ** 2))
            assert cupy.allclose(y, x1 + x2)

    @pytest.mark.thread_unsafe(reason="mutates global cache directory")
    def test_grid_sync_rawmodule(self, n, block):
        with use_temporary_cache_dir():
            mod_grid_sync = cupy.RawModule(
                code=_test_grid_sync, backend='nvcc',
                enable_cooperative_groups=True)
            x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
            x2 = cupy.ones((n, n), dtype='float32')
            y = cupy.zeros((n, n), dtype='float32')
            kern = mod_grid_sync.get_function('test_grid_sync')
            grid = (n * n + block - 1) // block
            kern((grid,), (block,), (x1, x2, y, n ** 2))
            assert cupy.allclose(y, x1 + x2)


_test_script = r'''
import pickle
import sys

import cupy as cp


N = 100
a = cp.random.random(N, dtype=cp.float32)
b = cp.random.random(N, dtype=cp.float32)
c = cp.empty_like(a)
with open('raw.pkl', 'rb') as f:
    ker = pickle.load(f)

if len(sys.argv) == 2:
    ker = ker.get_function(sys.argv[1])

ker((1,), (100,), (a, b, c, N))
assert cp.allclose(a + b, c)
assert ker.enable_cooperative_groups
'''


# Pickling/unpickling a RawModule should always success, whereas
# pickling/unpickling a RawKernel would fail if we don't enforce
# recompiling after unpickling it.
@pytest.mark.parametrize("compile", [
    pytest.param(False, id="nocompile"),
    pytest.param(True, id="compile"),
])
@pytest.mark.parametrize("raw", ['ker', 'mod', 'mod_ker'])
@pytest.mark.skipif(
    60 > int(cupy.cuda.device.get_compute_capability()),
    reason="Requires compute capability 6.0 or later")
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip,
    reason="HIP does not support enable_cooperative_groups")
class TestRawPicklable:
    @pytest.fixture(autouse=True)
    def configure(self, compile, raw):
        # test if kw-only arguments are properly handled or not
        ker, mod = None, None
        if raw == 'ker':
            ker = cupy.RawKernel(_test_source1, 'test_sum',
                                 backend='nvcc',
                                 enable_cooperative_groups=True)
        else:
            mod = cupy.RawModule(code=_test_source1,
                                 backend='nvcc',
                                 enable_cooperative_groups=True)

        with use_temporary_cache_dir() as temp_dir:
            yield compile, raw, ker, mod, temp_dir

    def _helper(self, raw, ker, mod):
        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)

        if raw == 'ker':
            impl = ker
        else:
            impl = mod.get_function('test_sum')
        impl((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(x1 + x2, y)

    def test_raw_picklable(self, configure):
        compile, raw, ker, mod, temp_dir = configure
        # force compiling before pickling
        if compile:
            self._helper(raw, ker, mod)

        if raw == 'ker':
            # pickle the RawKernel
            obj = ker
        elif raw == 'mod':
            # pickle the RawModule
            obj = mod
        elif raw == 'mod_ker':
            # pickle the RawKernel fetched from the RawModule
            obj = mod.get_function('test_sum')
        with open(temp_dir + '/raw.pkl', 'wb') as f:
            pickle.dump(obj, f)

        # dump test script to temp dir
        with open(temp_dir + '/TestRawPicklable.py', 'w') as f:
            f.write(_test_script)
        test_args = ['test_sum'] if raw == 'mod' else []

        # run another process to check the pickle
        s = subprocess.run([sys.executable, 'TestRawPicklable.py'] + test_args,
                           cwd=temp_dir)
        s.check_returncode()  # raise if unsuccessful


# a slightly more realistic kernel involving std utilities
std_code = r'''
#include <type_traits>

template<typename T,
         typename = typename std::enable_if<std::is_integral<T>::value>::type>
__global__ void shift (T* a, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        a[tid] += 100;
    }
}
'''


# Recent CCCL has made Jitify cold-launch very slow, see the discussion
# starting https://github.com/cupy/cupy/pull/8899#issuecomment-2613022424.
@testing.slow
@pytest.mark.parametrize("jitify", [
    pytest.param(
        False,
        marks=[
            pytest.mark.filterwarnings(
                "ignore:The jitify argument is deprecated:DeprecationWarning")
        ],
        id="no-jitify"),
    pytest.param(
        True,
        marks=[
            pytest.mark.thread_unsafe(
                reason=(
                    "Jitify seems to have problems, ",
                    "skip as largely unmaintained."
                )
            ),
            pytest.mark.filterwarnings(
                "ignore:jitify=True is deprecated:DeprecationWarning")
        ],
        id="jitify"),
])
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip,
    reason="Jitify does not support ROCm/HIP")
class TestRawJitify:
    @pytest.fixture(autouse=True)
    def configure(self):
        with use_temporary_cache_dir() as temp_dir:
            yield temp_dir

    def _helper(self, jitify, header, options=()):
        code = header
        code += _test_source1
        mod1 = cupy.RawModule(code=code,
                              backend='nvrtc',
                              options=options,
                              jitify=jitify)

        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)
        ker = mod1.get_function('test_sum')
        ker((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(x1 + x2, y)

    def _helper2(self, jitify, type_str):
        mod2 = cupy.RawModule(code=std_code,
                              jitify=jitify,
                              name_expressions=('shift<%s>' % type_str,))
        ker = mod2.get_function('shift<%s>' % type_str)
        N = 256
        a = cupy.random.random_integers(0, 7, N).astype(cupy.int32)
        b = a.copy()
        ker((1,), (N,), (a, N))
        assert cupy.allclose(a, b+100)

    def test_jitify1(self, jitify):
        # simply prepend an unused header
        hdr = '#include <cub/block/block_reduce.cuh>\n'
        # Starting CUDA 12.2, fp16/bf16 headers are intertwined, but due to
        # license issue we can't yet bundle bf16 headers. CUB offers us a
        # band-aid solution to avoid including the latter (NVIDIA/cub#478,
        # nvbugs 3641496).
        options = ('-DCUB_DISABLE_BF16_SUPPORT',)

        # Compiling CUB headers now works with or without Jitify.
        self._helper(jitify, hdr, options)

    def test_jitify2(self, jitify):
        # NVRTC cannot compile any code involving std
        if jitify:
            # Jitify will make it work
            self._helper2(jitify, 'int')
        else:
            with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
                self._helper2(jitify, 'int')
            assert 'cannot open source file' in str(ex.value)

    def test_jitify3(self, jitify):
        # We supply a type impossible to specialize. Jitify is still able to
        # locate the headers, but when it comes to the actual compilation,
        # NVRTC fails (raising the same exception) with different error
        # messages.
        ex_type = cupy.cuda.compiler.CompileException
        with pytest.raises(ex_type) as ex:
            self._helper2(jitify, 'float')
        if jitify:
            assert 'Error in parsing name expression' in str(ex.value)
        else:
            assert 'cannot open source file' in str(ex.value)

    def test_jitify4(self, jitify):
        # ensure JitifyException is raised with a broken code
        code = r'''
        __global__ void i_am_broken() {
        '''

        if jitify:
            ex_type = cupy.cuda.compiler.JitifyException
        else:
            ex_type = cupy.cuda.compiler.CompileException

        with pytest.raises(ex_type):
            mod = cupy.RawModule(code=code, jitify=jitify)
            ker = mod.get_function('i_am_broken')  # noqa
        # if Jitify could redirect its output, we would be able to check
        # the error log here as well (NVIDIA/jitify#79)

    def test_jitify5(self, configure, jitify):
        temp_dir = configure
        # If including a header that does not exist, Jitify would attempt to
        # comment it out and proceed. If this header is actually unused, then
        # everything would run just fine.

        hdr = 'I_INCLUDE_SOMETHING.h'
        with open(temp_dir + '/' + hdr, 'w') as f:
            dummy = '#include <cupy/I_DO_NOT_EXIST_WAH_HA_HA.h>\n'
            f.write(dummy)
        hdr = '#include "' + hdr + '"\n'

        if jitify:
            # Jitify would print a warning "[jitify] File not found" to stdout,
            # but as mentioned above and elsewhere, we can't capture it.
            self._helper(jitify, hdr, options=('-I'+temp_dir,))
        else:
            with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
                self._helper(jitify, hdr, options=(f"-I{temp_dir}",))
            assert 'cannot open source file' in str(ex.value)


@pytest.mark.parametrize("jitify,match", [
    (True, ".*"),
    (False, "Avoid passing.*jitify=False")
])
@pytest.mark.skipif(
    cupy.cuda.runtime.is_hip,
    reason="Jitify does not support ROCm/HIP")
@testing.slow
@pytest.mark.thread_unsafe(reason="uses temporary cache dir")
@use_temporary_cache_dir()
def test_jitify_deprecation_warning(jitify, match):
    with pytest.warns(DeprecationWarning, match=match):
        cupy.RawKernel(
            _test_source1, 'test_sum',
            backend='nvrtc', jitify=jitify)

    with pytest.warns(DeprecationWarning, match=match):
        cupy.RawModule(
            code=_test_source1,
            backend='nvrtc', jitify=jitify)

    # Not technically part of the rawkernel, but test warning in compile here:
    with pytest.warns(DeprecationWarning, match=match):
        compiler.compile_using_nvrtc("", options=(), jitify=jitify)
