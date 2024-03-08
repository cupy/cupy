import cupy
from cupyx import jit

"""
Test to ensure that this file can be imported without CUDA Toolkit.
"""


@cupy.memoize()
def user_func(a: cupy.ndarray):
    a.sum()


squared_diff = cupy.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = (x - y) * (x - y)',
    'squared_diff')


l2norm_kernel = cupy.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

complex_kernel = cupy.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void my_func(const complex<float>* x1, const complex<float>* x2,
             complex<float>* y, float a) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + a * x2[tid];
}
''', 'my_func')


@jit.rawkernel()
def elementwise_copy(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        y[i] = x[i]


cupy.show_config(_full=True)
