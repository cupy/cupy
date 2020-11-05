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
#kernel=cupy.RawModule(code=code,backend='nvcc')
#code = kernel.get_function("xyzw_frequency_thrust_device")
#
#in_str = 'xxxzzzwwax'
#count = cupy.zeros([1],dtype=cupy.int64)
#in_arr = cupy.array([ord(x) for x in in_str],dtype=cupy.int8)
#
#code(grid=(N,),block=(N,),args=(count,in_arr,len(in_str))) # count[0] == 9


#code2, opt2, dict2 = jitify.jitify_source(code, ('-arch=sm_75', '-I/usr/local/cuda/include'), {})
#code2, opt2, dict2 = jitify.jitify_source(code, ('-arch=sm_75', '-I/usr/local/cuda/include'), dict2)
code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '--include-path=/usr/local/cuda-10.2/include'), {})
code2, opt2, dict2 = jitify.jitify(code, ('-arch=compute_75', '--include-path=/usr/local/cuda-10.2/include'), dict2)
print(code2)
print(opt2)
for k, v in dict2.items():
    print(k, v)
