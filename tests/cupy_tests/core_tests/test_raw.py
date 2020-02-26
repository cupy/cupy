import os
import pytest
import shutil
import tempfile
import unittest

import cupy
from cupy import testing
from cupy.cuda import compiler


_test_source1 = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
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

if 'CUPY_CACHE_DIR' in os.environ:
    _old_cache_dir = os.environ['CUPY_CACHE_DIR']
    _is_cache_env_var_set = True
else:
    _old_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')
    _is_cache_env_var_set = False
_test_cache_dir = None


@testing.parameterize(*testing.product({
    'backend': ('nvrtc', 'nvcc'),
}))
class TestRaw(unittest.TestCase):

    def setUp(self):
        global _test_cache_dir
        _test_cache_dir = tempfile.mkdtemp()
        os.environ['CUPY_CACHE_DIR'] = _test_cache_dir

        self.kern = cupy.RawKernel(
            _test_source1, 'test_sum',
            backend=self.backend)
        self.mod2 = cupy.RawModule(
            code=_test_source2,
            backend=self.backend)
        self.mod3 = cupy.RawModule(
            code=_test_source3,
            options=('-DPRECISION=2',),
            backend=self.backend)

    def tearDown(self):
        # To avoid cache interference, we remove cached files after every test,
        # and restore users' old setting
        global _test_cache_dir
        shutil.rmtree(_test_cache_dir)
        if _is_cache_env_var_set:
            os.environ['CUPY_CACHE_DIR'] = _old_cache_dir
        else:
            os.environ.pop('CUPY_CACHE_DIR')
        compiler._empty_file_preprocess_cache = {}

    def _helper(self, kernel, dtype):
        N = 10
        x1 = cupy.arange(N**2, dtype=dtype).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=dtype)
        y = cupy.zeros((N, N), dtype=dtype)
        kernel((N,), (N,), (x1, x2, y, N**2))
        return x1, x2, y

    def test_basic(self):
        x1, x2, y = self._helper(self.kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

    def test_kernel_attributes(self):
        attrs = self.kern.attributes
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
        assert self.kern.num_regs > 0
        assert self.kern.max_threads_per_block > 0
        assert self.kern.shared_size_bytes == 0

    def test_module(self):
        module = self.mod2
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        x1, x2, y = self._helper(ker_sum, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

        x1, x2, y = self._helper(ker_times, cupy.float32)
        assert cupy.allclose(y, x1 * x2)

    def test_compiler_flag(self):
        module = self.mod3
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        x1, x2, y = self._helper(ker_sum, cupy.float64)
        assert cupy.allclose(y, x1 + x2)

        x1, x2, y = self._helper(ker_times, cupy.float64)
        assert cupy.allclose(y, x1 * x2)

    def test_invalid_compiler_flag(self):
        with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
            cupy.RawModule(
                code=_test_source3,
                options=('-DPRECISION=3',),
                backend=self.backend)
        assert 'precision not supported' in str(ex.value)

    def test_module_load_failure(self):
        # in principle this test is better done in test_driver.py, but
        # this error is more likely to appear when using RawModule, so
        # let us do it here
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            cupy.RawModule(
                path=os.path.expanduser('~/this_does_not_exist.cubin'),
                backend=self.backend)
        assert 'CUDA_ERROR_FILE_NOT_FOUND' in str(ex.value)

    def test_module_neither_code_nor_path(self):
        with pytest.raises(TypeError):
            cupy.RawModule()

    def test_module_both_code_and_path(self):
        with pytest.raises(TypeError):
            cupy.RawModule(
                code=_test_source1,
                path='test.cubin')

    def test_get_function_failure(self):
        # in principle this test is better done in test_driver.py, but
        # this error is more likely to appear when using RawModule, so
        # let us do it here
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            self.mod2.get_function('no_such_kernel')
        assert 'CUDA_ERROR_NOT_FOUND' in str(ex.value)

    def test_dynamical_parallelism(self):
        ker = cupy.RawKernel(_test_source4, 'test_kernel', options=('-dc',),
                             backend=self.backend)
        N = 169
        inner_chunk = 13
        x = cupy.zeros((N,), dtype=cupy.float32)
        ker((1,), (N//inner_chunk,), (x, N, inner_chunk))
        assert (x == 1.0).all()

    def test_dynamical_parallelism_compile_failure(self):
        # no option for separate compilation is given should cause an error
        ker = cupy.RawKernel(_test_source4, 'test_kernel',
                             backend=self.backend)
        N = 10
        inner_chunk = 2
        x = cupy.zeros((N,), dtype=cupy.float32)
        if self.backend == 'nvrtc':
            # raised when calling ls.complete()
            with pytest.raises(cupy.cuda.driver.CUDADriverError):
                ker((1,), (N//inner_chunk,), (x, N, inner_chunk))
        else:  # nvcc
            with pytest.raises(cupy.cuda.compiler.CompileException):
                ker((1,), (N//inner_chunk,), (x, N, inner_chunk))

    def test_cuFloatComplex(self):
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex64

        mod = cupy.RawModule(
            code=_test_cuComplex,
            translate_cucomplex=True)
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
        assert (out == a * b).all()

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
        assert (out == a * b + c).all()

        ker = mod.get_function('test_makef')
        ker((grid,), (block,), (out,))
        # because of precision issue, the (A==B).all() semantics would fail
        assert cupy.allclose(out, 1.8 - 1j * 8.7)

        ker = mod.get_function('test_upcast')
        ker((grid,), (block,), (a, out_up))
        assert (out_up == a.astype(cupy.complex128)).all()

    def test_cuDoubleComplex(self):
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex128

        mod = cupy.RawModule(
            code=_test_cuComplex,
            translate_cucomplex=True)
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
        assert (out == a * b).all()

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
        assert (out == a * b + c).all()

        ker = mod.get_function('test_make')
        ker((grid,), (block,), (out,))
        assert (out == 1.8 - 1j * 8.7).all()

        ker = mod.get_function('test_downcast')
        ker((grid,), (block,), (a, out_down))
        assert (out_down == a.astype(cupy.complex64)).all()

    def test_const_memory(self):
        mod = cupy.RawModule(code=test_const_mem, backend=self.backend)
        ker = mod.get_function('multiply_by_const')
        mem_ptr = mod.get_global('some_array')
        const_arr = cupy.ndarray((100,), cupy.float32, mem_ptr)
        data = cupy.arange(100, dtype=cupy.float32)
        const_arr[...] = data
        output_arr = cupy.ones(100, dtype=cupy.float32)
        ker((1,), (100,), (output_arr, cupy.int32(100)))
        assert (data == output_arr).all()


_test_grid_sync = r'''
#include <cooperative_groups.h>

extern "C" __global__
void test_grid_sync(const float* x1, const float* x2, float* y) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid];
    cg::sync(grid);
    y[size - tid - 1] += x2[size - tid - 1];
}
'''


@testing.parameterize(*testing.product({
    'n': [10, 100, 256]
}))
@unittest.skipUnless(
    9000 <= cupy.cuda.runtime.runtimeGetVersion(),
    'Requires CUDA 9.x or later')
@unittest.skipUnless(
    60 <= int(cupy.cuda.device.get_compute_capability()),
    'Requires compute capability 6.0 or later')
class TestRawGridSync(unittest.TestCase):

    def setUp(self):
        global _test_cache_dir
        _test_cache_dir = tempfile.mkdtemp()
        os.environ['CUPY_CACHE_DIR'] = _test_cache_dir

        self.kern_grid_sync = cupy.RawKernel(
            _test_grid_sync, 'test_grid_sync', backend='nvcc',
            enable_cooperative_groups=True)
        self.mod_grid_sync = cupy.RawModule(
            code=_test_grid_sync, backend='nvcc',
            enable_cooperative_groups=True)

    def tearDown(self):
        # To avoid cache interference, we remove cached files after every test,
        # and restore users' old setting
        global _test_cache_dir
        shutil.rmtree(_test_cache_dir)
        if _is_cache_env_var_set:
            os.environ['CUPY_CACHE_DIR'] = _old_cache_dir
        else:
            os.environ.pop('CUPY_CACHE_DIR')
        compiler._empty_file_preprocess_cache = {}

    def test_grid_sync_rawkernel(self):
        n = self.n
        x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
        x2 = cupy.ones((n, n), dtype='float32')
        y = cupy.zeros((n, n), dtype='float32')
        self.kern_grid_sync((n,), (n,), (x1, x2, y, n ** 2))
        assert cupy.allclose(y, x1 + x2)

    def test_grid_sync_rawmodule(self):
        n = self.n
        x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
        x2 = cupy.ones((n, n), dtype='float32')
        y = cupy.zeros((n, n), dtype='float32')
        kern = self.mod_grid_sync.get_function('test_grid_sync')
        kern((n,), (n,), (x1, x2, y, n ** 2))
        assert cupy.allclose(y, x1 + x2)
