import os
import sys
import time

import cupy  # noqa: F401  (L3: full import must succeed)
from cupy.backends.ascend import bisheng

code = r'''
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void tiny_copy(
        GM_ADDR z, GM_ADDR x, uint64_t n) {
    GlobalTensor<float> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    zGm.SetGlobalBuffer((__gm__ float*)z);
    uint64_t i = GetBlockIdx();
    if (i < n) {
        zGm.SetValue(i, xGm.GetValue(i));
    }
}
'''

print('--- 1. bisheng compile_module_with_cache (first, JIT) ---')
t0 = time.time()
fatbin = bisheng.compile_module_with_cache(code)
print(f'compiled OK: {len(fatbin)} bytes in {time.time() - t0:.1f}s')

print('--- 2. second call must hit the disk cache ---')
t0 = time.time()
fatbin2 = bisheng.compile_module_with_cache(code)
assert fatbin == fatbin2
print(f'cache hit OK: {time.time() - t0:.3f}s')

print('--- 3. cache dir ---')
d = bisheng.kernel_cache_dir()
print(d, sorted(os.listdir(d)))

print('--- 4. corrupted cache entry -> recompile ---')
name = bisheng._cache_key(code, (), bisheng.default_soc())
with open(os.path.join(d, name), 'r+b') as f:
    f.seek(50)
    f.write(b'\x00\x00\x00\x00')
t0 = time.time()
fatbin3 = bisheng.compile_module_with_cache(code)
assert len(fatbin3) > 1000 and time.time() - t0 > 0.5  # recompiled, not a hit
# and the cache is healthy again (instant hit)
t0 = time.time()
fatbin4 = bisheng.compile_module_with_cache(code)
assert time.time() - t0 < 0.1
print('corruption recovery OK')

print('--- 5. compile error -> BishengCompileException ---')
try:
    bisheng.compile_module_with_cache('this is not C++')
except bisheng.BishengCompileException as e:
    print('exception OK:', str(e).splitlines()[0][:120])

print('--- 6. RawKernel object (full API surface) ---')
k = cupy.RawKernel(code, 'tiny_copy')
print('RawKernel:', k.name, k.backend, k.code is not None)
assert cupy.RawModule is not None
try:
    m = cupy.RawModule(code=code, name_expressions=['tiny_copy'])
    m.get_function('tiny_copy')
    raise AssertionError('should have raised')
except NotImplementedError:
    print('name_expressions rejected at use time OK')
try:
    cupy.RawModule(code=code, path='x')
except TypeError as e:
    print('code/path exclusivity OK')

print('--- 7. RawKernel launch path (no NPU: expect a clear failure) ---')
try:
    k((1,), (1,), (0, 0, 0))
    print('launch OK (unexpected without NPU!)')
except Exception as e:
    print(f'{type(e).__name__}: {str(e)[:160]}')

print('ALL SMOKE CHECKS DONE')
