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


def validateaxis(axis):
    if axis is not None:
        axis_type = type(axis)

        if axis_type == tuple:
            raise TypeError(
                'Tuples are not accepted for the \'axis\' '
                'parameter. Please pass in one of the '
                'following: {-2, -1, 0, 1, None}.')

        if not cupy.issubdtype(cupy.dtype(axis_type), cupy.integer):
            raise TypeError('axis must be an integer, not {name}'
                            .format(name=axis_type.__name__))

        if not (-2 <= axis <= 1):
            raise ValueError('axis out of range')
