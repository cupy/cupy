import functools
import warnings

import numpy

import cupy
from cupy import core

_correlate_kernel = core.RawKernel(r'''
  #define MAX_TPB 256
  extern "C"{
  __device__ double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

  __global__ void dot_kernel(const double* x1, const double* x2,
  double* y, int j, int j1, int j2, int n){
    __shared__ double temp[MAX_TPB];
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = x1[j1 + tid] * x2[j2 + tid];
    __syncthreads();

    if (threadIdx.x == 0)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += temp[i];
        }
        atomicAdd(&y[j], sum);
     }
  }
     __global__ void correlate_kernel(const double* x1, const double* x2,
     double* y, int j, int j1, int j2, int n, int mode){
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int nTPB = MAX_TPB;
        if (mode == 0){ // left
            nTPB = min (n + tid, nTPB);
            dot_kernel<<<(n + tid)/nTPB + 1, nTPB>>>
            (x1, x2, y, j + tid, j1, j2 - tid, nTPB);
        }else if (mode == 1){ //right
            nTPB = min (n - tid, nTPB);
            dot_kernel<<<(n - tid)/nTPB + 1, nTPB>>>
            (x1, x2, y, j + tid,  j1 + tid, j2, nTPB);
        }else{ // valid
            nTPB = min (n, nTPB);
            dot_kernel<<<n/nTPB + 1, nTPB>>>
            (x1, x2, y, j + tid,  j1 + tid, j2, nTPB);
        }
     }
}''', 'correlate_kernel', ('-rdc=true',))


def corrcoef(a, y=None, rowvar=True, bias=None, ddof=None):
    """Returns the Pearson product-moment correlation coefficients of an array.

    Args:
        a (cupy.ndarray): Array to compute the Pearson product-moment
            correlation coefficients.
        y (cupy.ndarray): An additional set of variables and observations.
        rowvar (bool): If ``True``, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is
            transposed.
        bias (None): Has no effect, do not use.
        ddof (None): Has no effect, do not use.

    Returns:
        cupy.ndarray: The Pearson product-moment correlation coefficients of
        the input array.

    .. seealso:: :func:`numpy.corrcoef`

    """
    if bias is not None or ddof is not None:
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning)

    out = cov(a, y, rowvar)
    try:
        d = cupy.diag(out)
    except ValueError:
        return out / out

    stddev = cupy.sqrt(d.real)
    out /= stddev[:, None]
    out /= stddev[None, :]

    cupy.clip(out.real, -1, 1, out=out.real)
    if cupy.iscomplexobj(out):
        cupy.clip(out.imag, -1, 1, out=out.imag)

    return out


def correlate(a, v, mode='valid'):
    """Returns the cross-correlation of two 1-dimensional sequences.

    Args:
        a (cupy.ndarray): first 1-dimensional input.
        v (cupy.ndarray): second 1-dimensional input.
        mode (optional): `valid`, `same`, `full`

    Returns:
        cupy.ndarray: Discrete cross-correlation of a and v.

    .. seealso:: :func:`numpy.correlate`

    """
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("object too deep for desired array")
    if cupy.iscomplexobj(v):
        raise NotImplementedError(
            "Complex 1D correlation is not supported currently")
    inverted, output = correlate_util(a, v, mode)
    if inverted:
        output = output[::-1]
    return output


def correlate_util(a1, a2, mode):
    inverted = 0
    max_threads = 1024
    dtype = a1.dtype
    if a1.size == 0 or a2.size == 0:
        raise ValueError("Array arguments cannot be empty")
    if a1.size < a2.size:
        a1, a2 = a2, a1
        inverted = 1
    length = n1 = a1.size
    n = n2 = a2.size
    left, right, length = _generate_boundaries(mode, length, n)
    output = cupy.zeros(shape=length, dtype=float)
    a1 = cupy.ascontiguousarray(a1, float)
    a2 = cupy.ascontiguousarray(a2, float)
    if left:
        nTPB = min(left, max_threads)
        _correlate_kernel((int((left + nTPB - 1) / nTPB),), (nTPB,),
                          (a1, a2, output, 0, 0, left, n - left, 0))
    nTPB = min(n1 - n2 + 1, max_threads)
    _correlate_kernel((int((n1 - n2 + nTPB) / nTPB),), (nTPB,),
                      (a1, a2, output, left, 0, 0, n, 2))
    if right:
        nTPB = min(right, max_threads)
        _correlate_kernel((int((right + nTPB - 1) / nTPB),), (nTPB,),
                          (a1, a2, output, left + n1 - n2 + 1,
                           n1 - n2 + 1, 0, n, 1))
    return inverted, output.astype(dtype=dtype, copy=False)


def _generate_boundaries(mode, length, n):
    if mode == 'valid':
        length += 1 - n
        left = right = 0
    elif mode == 'same':
        left = int(n / 2)
        right = n - left - 1
    elif mode == 'full':
        left = right = n - 1
        length += n - 1
    else:
        raise ValueError("Invalid mode")
    return left, right, length


def cov(a, y=None, rowvar=True, bias=False, ddof=None):
    """Returns the covariance matrix of an array.

    This function currently does not support ``fweights`` and ``aweights``
    options.

    Args:
        a (cupy.ndarray): Array to compute covariance matrix.
        y (cupy.ndarray): An additional set of variables and observations.
        rowvar (bool): If ``True``, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is
            transposed.
        bias (bool): If ``False``, normalization is by ``(N - 1)``, where N is
            the number of observations given (unbiased estimate). If ``True``,
            then normalization is by ``N``.
        ddof (int): If not ``None`` the default value implied by bias is
            overridden. Note that ``ddof=1`` will return the unbiased estimate
            and ``ddof=0`` will return the simple average.

    Returns:
        cupy.ndarray: The covariance matrix of the input array.

    .. seealso:: :func:`numpy.cov`

    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')

    if a.ndim > 2:
        raise ValueError('Input must be <= 2-d')

    if y is None:
        dtype = numpy.promote_types(a.dtype, numpy.float64)
    else:
        if y.ndim > 2:
            raise ValueError('y must be <= 2-d')
        dtype = functools.reduce(numpy.promote_types,
                                 (a.dtype, y.dtype, numpy.float64))

    X = cupy.array(a, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return cupy.array([]).reshape(0, 0)
    if y is not None:
        y = cupy.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = core.concatenate_method((X, y), axis=0)

    if ddof is None:
        ddof = 0 if bias else 1

    fact = X.shape[1] - ddof
    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice',
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= X.mean(axis=1)[:, None]
    out = X.dot(X.T.conj()) * (1 / cupy.float64(fact))

    return out.squeeze()
