import cupy as cp


a = cp.empty(100)
cp.cuda.Device().synchronize()
ker = r"""
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
extern "C" __global__ void test_grid() {

  cg::thread_block y = cg::this_thread_block();
  y.sync();
}
"""
myker = cp.RawKernel(ker, 'test_grid', options=('-std=c++11',))
myker((1,), (256,), ())
cp.cuda.Device().synchronize()
