import cupy
import cupy.sparse.base


_preamble_atomic_add = '''
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long* address_as_ull =
                                          (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
'''


def isintlike(x):
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isscalarlike(x):
    return cupy.isscalar(x) or (cupy.sparse.base.isdense(x) and x.ndim == 0)


def isshape(x):
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    return isintlike(m) and isintlike(n)
