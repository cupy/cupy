import numpy
import six

import cupy
from cupy import core


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    """
    # TODO(okuta): check type
    return a.sum(axis, dtype, out, keepdims)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    """
    # TODO(okuta): check type
    return a.prod(axis, dtype, out, keepdims)


# TODO(okuta): Implement nansum


def _axis_to_first(x, axis):
    if axis < 0:
        axis = x.ndim + axis
    trans = [axis] + [a for a in six.moves.range(x.ndim) if a != axis]
    pre = list(six.moves.range(1, axis + 1))
    succ = list(six.moves.range(axis + 1, x.ndim))
    revert = pre + [0] + succ
    return trans, revert


def _proc_as_batch(proc, x, axis):
    if x.shape[axis] == 0:
        return cupy.empty_like(x)
    trans, revert = _axis_to_first(x, axis)
    t = x.transpose(trans)
    s = t.shape
    r = t.reshape(x.shape[axis], -1)
    pos = 1
    size = r.size
    batch = r.shape[1]
    while pos < size:
        proc(pos, batch, r, size=size)
        pos <<= 1
    return r.reshape(s).transpose(revert)


def _cum_core(a, axis, dtype, out, kern, batch_kern):
    if out is None:
        if dtype is None:
            kind = a.dtype.kind
            if kind == 'b':
                dtype = numpy.dtype('l')
            elif kind == 'i' and a.dtype.itemsize < numpy.dtype('l').itemsize:
                dtype = numpy.dtype('l')
            elif kind == 'u' and a.dtype.itemsize < numpy.dtype('L').itemsize:
                dtype = numpy.dtype('L')
            else:
                dtype = a.dtype

        out = a.astype(dtype)
    else:
        out[...] = a

    if axis is None:
        out = out.ravel()
    elif not (-a.ndim <= axis < a.ndim):
        raise core.core._AxisError('axis(={}) out of bounds'.format(axis))
    else:
        return _proc_as_batch(batch_kern, out, axis=axis)

    pos = 1
    while pos < out.size:
        kern(pos, out, size=out.size)
        pos <<= 1
    return out


_cumsum_batch_kern = core.ElementwiseKernel(
    'int64 pos, int64 batch', 'raw T x',
    '''
    ptrdiff_t b = i % batch;
    ptrdiff_t j = i / batch;
    if (j & pos) {
      const ptrdiff_t dst_index[] = {j, b};
      const ptrdiff_t src_index[] = {j ^ pos | (pos - 1), b};
      x[dst_index] += x[src_index];
    }
    ''',
    'cumsum_batch_kernel'
)
_cumsum_kern = core.ElementwiseKernel(
    'int64 pos', 'raw T x',
    '''
    if (i & pos) {
      x[i] += x[i ^ pos | (pos - 1)];
    }
    ''',
    'cumsum_kernel'
)


def cumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumsum`

    """
    return _cum_core(a, axis, dtype, out, _cumsum_kern, _cumsum_batch_kern)


_cumprod_batch_kern = core.ElementwiseKernel(
    'int64 pos, int64 batch', 'raw T x',
    '''
    ptrdiff_t b = i % batch;
    ptrdiff_t j = i / batch;
    if (j & pos) {
      const ptrdiff_t dst_index[] = {j, b};
      const ptrdiff_t src_index[] = {j ^ pos | (pos - 1), b};
      x[dst_index] *= x[src_index];
    }
    ''',
    'cumprod_batch_kernel'
)
_cumprod_kern = core.ElementwiseKernel(
    'int64 pos', 'raw T x',
    '''
    if (i & pos) {
      x[i] *= x[i ^ pos | (pos - 1)];
    }
    ''',
    'cumprod_kernel'
)


def cumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumprod`

    """
    return _cum_core(a, axis, dtype, out, _cumprod_kern, _cumprod_batch_kern)

def diff(a, n=1, axis=-1):
  """Calculate the n-th discrete difference along the given axis.
  Args:
      a (cupy.ndarray): Input array.
      n (int): The number of times values are differenced. If zero, the input
                  is returned as-is.
      axis (int): The axis along which the difference is taken, default is the
                    last axis.
  Returns:
      cupy.ndarray: The result array.
  .. seealso:: :func:`numpy.diff`
  """
  
  if n == 0:
      return a
  if n < 0:
      raise ValueError(
          "order must be non-negative but got " + repr(n))

  a = cupy.asanyarray(a)
  nd = a.ndim
  
  slice1 = [slice(None)] * nd
  slice2 = [slice(None)] * nd
  slice1[axis] = slice(1, None)
  slice2[axis] = slice(None, -1)
  slice1 = tuple(slice1)
  slice2 = tuple(slice2)

  op = not_equal if a.dtype == numpy.bool_ else cupy.subtract
  for _ in range(n):
      a = op(a[slice1], a[slice2])

  return a

def unwrap(p, discont=numpy.pi, axis=-1):
  """Unwrap by changing deltas between values to 2*pi complement.
  Args:
      p (cupy.ndarray): Input array.
      discont (float): Maximum discontinuity between values, default is ``pi``.
      axis (int): Axis along which unwrap will operate, default is the last axis.
  Returns:
      cupy.ndarray: The result array.
  .. seealso:: :func:`numpy.unwrap`
  """
  
  p = cupy.asarray(p)
  nd = p.ndim
  dd = diff(p, axis=axis)
  slice1 = [slice(None, None)]*nd     # full slices
  slice1[axis] = slice(1, None)
  slice1 = tuple(slice1)
  ddmod = cupy.mod(dd + numpy.pi, 2*numpy.pi) - numpy.pi
  cupy.copyto(ddmod, numpy.pi, where=(ddmod == -numpy.pi) & (dd > 0))
  ph_correct = ddmod - dd
  cupy.copyto(ph_correct, 0, where=cupy.abs(dd) < discont)
  up = cupy.array(p, copy=True, dtype='d')
  up[slice1] = p[slice1] + cupy.cumsum(ph_correct, axis=axis)
  return up

# TODO(okuta): Implement diff


# TODO(okuta): Implement ediff1d


# TODO(okuta): Implement gradient


# TODO(okuta): Implement cross


# TODO(okuta): Implement trapz
