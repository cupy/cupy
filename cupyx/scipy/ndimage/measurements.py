import warnings

import numpy

import cupy
from cupy import core
from cupy import _util


def label(input, structure=None, output=None):
    """Labels features in an array.

    Args:
        input (cupy.ndarray): The input array.
        structure (array_like or None): A structuring element that defines
            feature connections. ```structure``` must be centersymmetric. If
            None, structure is automatically generated with a squared
            connectivity equal to one.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
    Returns:
        label (cupy.ndarray): An integer array where each unique feature in
        ```input``` has a unique label in the array.

        num_features (int): Number of features found.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.label`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = _generate_binary_structure(input.ndim, 1)
    elif isinstance(structure, cupy.ndarray):
        structure = cupy.asnumpy(structure)
    structure = numpy.array(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for i in structure.shape:
        if i != 3:
            raise ValueError('structure dimensions must be equal to 3')

    if isinstance(output, cupy.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = cupy.empty(input.shape, numpy.int32)
        else:
            output = cupy.empty(input.shape, output)

    if input.size == 0:
        # empty
        maxlabel = 0
    elif input.ndim == 0:
        # 0-dim array
        maxlabel = 0 if input.item() == 0 else 1
        output[...] = maxlabel
    else:
        if output.dtype != numpy.int32:
            y = cupy.empty(input.shape, numpy.int32)
        else:
            y = output
        maxlabel = _label(input, structure, y)
        if output.dtype != numpy.int32:
            output[...] = y[...]

    if caller_provided_output:
        return maxlabel
    else:
        return output, maxlabel


def _generate_binary_structure(rank, connectivity):
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity


def _label(x, structure, y):
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]
    dirs = cupy.array(dirs, dtype=numpy.int32)
    ndirs = indxs.shape[0]
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    count = cupy.zeros(2, dtype=numpy.int32)
    _kernel_init()(x, y)
    _kernel_connect()(y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    _kernel_count()(y, count, size=y.size)
    maxlabel = int(count[0])
    labels = cupy.empty(maxlabel, dtype=numpy.int32)
    _kernel_labels()(y, count, labels, size=y.size)
    _kernel_finalize()(maxlabel, cupy.sort(labels), y, size=y.size)
    return maxlabel


def _kernel_init():
    return core.ElementwiseKernel(
        'X x', 'Y y', 'if (x == 0) { y = -1; } else { y = i; }',
        'cupyx_nd_label_init')


def _kernel_connect():
    return core.ElementwiseKernel(
        'raw int32 shape, raw int32 dirs, int32 ndirs, int32 ndim',
        'raw Y y',
        '''
        if (y[i] < 0) continue;
        for (int dr = 0; dr < ndirs; dr++) {
            int j = i;
            int rest = j;
            int stride = 1;
            int k = 0;
            for (int dm = ndim-1; dm >= 0; dm--) {
                int pos = rest % shape[dm] + dirs[dm + dr * ndim];
                if (pos < 0 || pos >= shape[dm]) {
                    k = -1;
                    break;
                }
                k += pos * stride;
                rest /= shape[dm];
                stride *= shape[dm];
            }
            if (k < 0) continue;
            if (y[k] < 0) continue;
            while (1) {
                while (j != y[j]) { j = y[j]; }
                while (k != y[k]) { k = y[k]; }
                if (j == k) break;
                if (j < k) {
                    int old = atomicCAS( &y[k], k, j );
                    if (old == k) break;
                    k = old;
                }
                else {
                    int old = atomicCAS( &y[j], j, k );
                    if (old == j) break;
                    j = old;
                }
            }
        }
        ''',
        'cupyx_nd_label_connect')


def _kernel_count():
    return core.ElementwiseKernel(
        '', 'raw Y y, raw int32 count',
        '''
        if (y[i] < 0) continue;
        int j = i;
        while (j != y[j]) { j = y[j]; }
        if (j != i) y[i] = j;
        else atomicAdd(&count[0], 1);
        ''',
        'cupyx_nd_label_count')


def _kernel_labels():
    return core.ElementwiseKernel(
        '', 'raw Y y, raw int32 count, raw int32 labels',
        '''
        if (y[i] != i) continue;
        int j = atomicAdd(&count[1], 1);
        labels[j] = i;
        ''',
        'cupyx_nd_label_labels')


def _kernel_finalize():
    return core.ElementwiseKernel(
        'int32 maxlabel', 'raw int32 labels, raw Y y',
        '''
        if (y[i] < 0) {
            y[i] = 0;
            continue;
        }
        int yi = y[i];
        int j_min = 0;
        int j_max = maxlabel - 1;
        int j = (j_min + j_max) / 2;
        while (j_min < j_max) {
            if (yi == labels[j]) break;
            if (yi < labels[j]) j_max = j - 1;
            else j_min = j + 1;
            j = (j_min + j_max) / 2;
        }
        y[i] = j + 1;
        ''',
        'cupyx_nd_label_finalize')


_ndimage_variance_kernel = core.ElementwiseKernel(
    'T input, R labels, raw X index, uint64 size, raw float64 mean',
    'raw float64 out',
    """
    for (ptrdiff_t j = 0; j < size; j++) {
      if (labels == index[j]) {
        atomicAdd(&out[j], (input - mean[j]) * (input - mean[j]));
        break;
      }
    }
    """)


_ndimage_sum_kernel = core.ElementwiseKernel(
    'T input, R labels, raw X index, uint64 size',
    'raw float64 out',
    """
    for (ptrdiff_t j = 0; j < size; j++) {
      if (labels == index[j]) {
        atomicAdd(&out[j], input);
        break;
      }
    }
    """)


def _ndimage_sum_kernel_2(input, labels, index, sum_val, batch_size=4):
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape(
            (-1,) + (1,) * input.ndim)
        sum_axes = tuple(range(1, 1 + input.ndim))
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(
            axis=sum_axes)
    return sum_val


_ndimage_mean_kernel = core.ElementwiseKernel(
    'T input, R labels, raw X index, uint64 size',
    'raw float64 out, raw uint64 count',
    """
    for (ptrdiff_t j = 0; j < size; j++) {
      if (labels == index[j]) {
        atomicAdd(&out[j], input);
        atomicAdd(&count[j], 1);
        break;
      }
    }
    """)


def _ndimage_mean_kernel_2(input, labels, index, batch_size=4,
                           return_count=False):
    sum_val = cupy.empty_like(index, dtype=cupy.float64)
    count = cupy.empty_like(index, dtype=cupy.uint64)
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape(
            (-1,) + (1,) * input.ndim)
        mean_axes = tuple(range(1, 1 + input.ndim))
        count[i:i + batch_size] = matched.sum(axis=mean_axes)
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(
            axis=mean_axes)
    if return_count:
        return sum_val / count, count
    return sum_val / count


def _mean_driver(input, labels, index, return_count=False, use_kern=False):
    if use_kern:
        return _ndimage_mean_kernel_2(input, labels, index,
                                      return_count=return_count)

    out = cupy.zeros_like(index, cupy.float64)
    count = cupy.zeros_like(index, dtype=cupy.uint64)
    sum, count = _ndimage_mean_kernel(input,
                                      labels, index, index.size, out, count)
    if return_count:
        return sum / count, count
    return sum / count


def variance(input, labels=None, index=None):
    """Calculates the variance of the values of an n-D image array, optionally
    at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
        variance (cupy.ndarray): Values of variance, for each sub-region if
            `labels` and `index` are specified.

    .. seealso:: :func:`scipy.ndimage.variance`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')

    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError("cupyx.scipy.ndimage.variance doesn't support %{}"
                        "".format(input.dtype.type))

    use_kern = False
    # There is constraints on types because of atomicAdd() in CUDA.
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32,
                           cupy.float64, cupy.uint32, cupy.uint64,
                           cupy.ulonglong]:
        warnings.warn(
            'Using the slower implmentation as '
            'cupyx.scipy.ndimage.sum supports int32, float16, '
            'float32, float64, uint32, uint64 as data types'
            'for the fast implmentation', _util.PerformanceWarning)
        use_kern = True

    def calc_var_with_intermediate_float(input):
        vals_c = input - input.mean()
        count = vals_c.size
        # Does not use `ndarray.mean()` here to return the same results as
        # SciPy does, especially in case `input`'s dtype is float16.
        return cupy.square(vals_c).sum() / cupy.asanyarray(count).astype(float)

    if labels is None:
        return calc_var_with_intermediate_float(input)

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    input, labels = cupy.broadcast_arrays(input, labels)

    if index is None:
        return calc_var_with_intermediate_float(input[labels > 0])

    if cupy.isscalar(index):
        return calc_var_with_intermediate_float(input[labels == index])

    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return (input[labels == index]).var().astype(cupy.float64,
                                                         copy=False)

    mean_val, count = _mean_driver(input, labels, index, True, use_kern)
    if use_kern:
        new_axis = (..., *(cupy.newaxis for _ in range(input.ndim)))
        return cupy.where(labels[None, ...] == index[new_axis],
                          cupy.square(input - mean_val[new_axis]),
                          0).sum(tuple(range(1, input.ndim + 1))) / count
    out = cupy.zeros_like(index, dtype=cupy.float64)
    return _ndimage_variance_kernel(input, labels, index, index.size, mean_val,
                                    out) / count


def sum(input, labels=None, index=None):
    """Calculates the sum of the values of an n-D image array, optionally
       at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
       sum (cupy.ndarray): sum of values, for each sub-region if
       `labels` and `index` are specified.

    .. seealso:: :func:`scipy.ndimage.sum`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')

    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError("cupyx.scipy.ndimage.sum does not support %{}".format(
            input.dtype.type))

    use_kern = False
    # There is constraints on types because of atomicAdd() in CUDA.
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32,
                           cupy.float64, cupy.uint32, cupy.uint64,
                           cupy.ulonglong]:
        warnings.warn(
            'Using the slower implmentation as '
            'cupyx.scipy.ndimage.sum supports int32, float16, '
            'float32, float64, uint32, uint64 as data types'
            'for the fast implmentation', _util.PerformanceWarning)
        use_kern = True

    if labels is None:
        return input.sum()

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    input, labels = cupy.broadcast_arrays(input, labels)

    if index is None:
        return input[labels != 0].sum()

    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return (input[labels == index]).sum()

    if index.size == 0:
        return cupy.array([], dtype=cupy.int64)

    out = cupy.zeros_like(index, dtype=cupy.float64)

    # The following parameters for sum where determined using a Tesla P100.
    if (input.size >= 262144 and index.size <= 4) or use_kern:
        return _ndimage_sum_kernel_2(input, labels, index, out)
    return _ndimage_sum_kernel(input, labels, index, index.size, out)


def mean(input, labels=None, index=None):
    """Calculates the mean of the values of an n-D image array, optionally
       at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
        mean (cupy.ndarray): mean of values, for each sub-region if
        `labels` and `index` are specified.


    .. seealso:: :func:`scipy.ndimage.mean`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')

    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError("cupyx.scipy.ndimage.mean does not support %{}".format(
            input.dtype.type))

    use_kern = False
    # There is constraints on types because of atomicAdd() in CUDA.
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32,
                           cupy.float64, cupy.uint32, cupy.uint64,
                           cupy.ulonglong]:
        warnings.warn(
            'Using the slower implmentation as '
            'cupyx.scipy.ndimage.mean supports int32, float16, '
            'float32, float64, uint32, uint64 as data types '
            'for the fast implmentation', _util.PerformanceWarning)
        use_kern = True

    def calc_mean_with_intermediate_float(input):
        sum = input.sum()
        count = input.size
        # Does not use `ndarray.mean()` here to return the same results as
        # SciPy does, especially in case `input`'s dtype is float16.
        return sum / cupy.asanyarray(count).astype(float)

    if labels is None:
        return calc_mean_with_intermediate_float(input)

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    input, labels = cupy.broadcast_arrays(input, labels)

    if index is None:
        return calc_mean_with_intermediate_float(input[labels > 0])

    if cupy.isscalar(index):
        return calc_mean_with_intermediate_float(input[labels == index])

    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return (input[labels == index]).mean(dtype=cupy.float64)

    return _mean_driver(input, labels, index, use_kern=use_kern)


def standard_deviation(input, labels=None, index=None):
    """Calculates the standard deviation of the values of an n-D image array,
    optionally at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
        standard_deviation (cupy.ndarray): standard deviation of values, for
        each sub-region if `labels` and `index` are specified.

    .. seealso:: :func:`scipy.ndimage.standard_deviation`
    """
    return cupy.sqrt(variance(input, labels, index))
