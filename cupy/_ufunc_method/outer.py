import cupy


def ufunc_outer(ufunc, A, B, **kwargs):
    A = cupy.asarray(A)
    B = cupy.asarray(B)
    ndim_a = A.ndim
    ndim_b = B.ndim
    A = A.reshape(A.shape + (1,) * ndim_b)
    B = B.reshape((1,) * ndim_a + B.shape)
    return ufunc(A, B, **kwargs)
