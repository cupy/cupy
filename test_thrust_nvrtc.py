import cupy
import cupy.cuda.jitify as jitify


N = 100
# TODO: replace the first line (my_program\n) by hash!
code = """my_program_ohoh
#include <thrust/count.h>

extern "C" __global__
void xyzw_frequency_thrust_device(int *count, char *text, int n)
{
  const char letters[] { 'x','y','z','w' };

  *count = thrust::count_if(thrust::device, text, text+n, [=](char c) {
    for (const auto x : letters) 
      if (c == x) return true;
    return false;
  });
}"""

code = """my_program_omg
//#include <thrust/reduce.h>
//#include <thrust/execution_policy.h>

__global__ void test2(const float * __restrict__ d_data, float * __restrict__ d_results, const int Nrows, const int Ncols) {

    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < Nrows) {
        d_results[tid] = thrust::reduce(thrust::device, d_data + tid * Ncols, d_data + (tid + 1) * Ncols);
    }

}
"""

with open("/home/leofang/.cupy/kernel_cache_cuda10.2/3240783440133f52312c6a239d72162d_2.cubin.cu") as f:
    code = f.read()
    code = 'my_cub_program\n' + code
#kernel=cupy.RawModule(code=code,backend='nvcc')
#code = kernel.get_function("xyzw_frequency_thrust_device")
#
#in_str = 'xxxzzzwwax'
#count = cupy.zeros([1],dtype=cupy.int64)
#in_arr = cupy.array([ord(x) for x in in_str],dtype=cupy.int8)
#
#code(grid=(N,),block=(N,),args=(count,in_arr,len(in_str))) # count[0] == 9


# Use --include-path to avoid problems, see https://github.com/NVIDIA/jitify/issues/65
#code2, opt2, dict2 = jitify.jitify_source(code, ('-arch=sm_75', '-I/usr/local/cuda/include'), {})
#code2, opt2, dict2 = jitify.jitify_source(code, ('-arch=sm_75', '-I/usr/local/cuda/include'), dict2)

#print(code)
#code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '--include-path=/usr/local/cuda-10.2/include'), {})
#code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '--include-path=/usr/local/cuda-10.2/include'), dict2)

code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '-I/home/leofang/dev/cupy_cuda10.2/cupy/core/include/'))
#code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '-I/home/leofang/dev/cupy_cuda10.2/cupy/core/include/'))
print(code2)
print(opt2)
print(len(dict2))
#for k, v in dict2.items():
#    #print(k, v)
#    print(k)
