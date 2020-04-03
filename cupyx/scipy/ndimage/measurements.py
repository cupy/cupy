import numpy

import cupy

import string


def label(input, structure=None, output=None):
    """Labels features in an array

    Args:
        input (cupy.ndarray): The input array.
        structure (array_like or None): A structuring element that defines \
            feature connections. ```structure``` must be centersymmetric. If \
            None, structure is automatically generated with a squared \
            connectivity equal to one.
        output (cupy.ndarray, dtype or None): The array in which to place the \
            output.

    Returns:
        label (cupy.ndarray): An integer array where each unique feature in \
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
    return cupy.ElementwiseKernel(
        'X x', 'Y y', 'if (x == 0) { y = -1; } else { y = i; }',
        'cupyx_nd_label_init')


def _kernel_connect():
    return cupy.ElementwiseKernel(
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
    return cupy.ElementwiseKernel(
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
    return cupy.ElementwiseKernel(
        '', 'raw Y y, raw int32 count, raw int32 labels',
        '''
        if (y[i] != i) continue;
        int j = atomicAdd(&count[1], 1);
        labels[j] = i;
        ''',
        'cupyx_nd_label_labels')


def _kernel_finalize():
    return cupy.ElementwiseKernel(
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


def _variance_kernel():
    input_arguments = 'T image, R label, raw X index, int32 size, raw Z mean'
    output_arguments = 'raw Z ret'
    _kernel_template = """
    for (ptrdiff_t j = 0; j < size; j++) {
      if (label == index[j]) {
        atomicAdd(&ret[j], (image - mean[j]) * (image - mean[j]));
        break;
      }
    }
    """
    return cupy.ElementwiseKernel(input_arguments, output_arguments,
                                  _kernel_template)


def _stats(input, labels=None, index=None, mean=False):

    input_arguments = 'T image, R label, raw X index, int32 size'
    _kernel_template = string.Template("""
    for (ptrdiff_t j = 0; j < size; j++) {
      if (label == index[j]) {
        atomicAdd(&ret[j], image);
        $mean
        break;
      }
    }
    """)
    if not mean:
        output_arguments = 'raw Y ret'
        _kernel_template = _kernel_template.substitute(mean="")
        ret = cupy.zeros_like(index).astype(input.dtype)
    else:
        output_arguments = 'raw Y ret, raw int32 count'
        _kernel_template = _kernel_template.substitute(
            mean="atomicAdd(&count[j], 1);")
        ret = cupy.zeros_like(index).astype(input.dtype)
        count = cupy.zeros_like(index).astype(cupy.int32)

    _kernel = cupy.ElementwiseKernel(input_arguments, output_arguments,
                                     _kernel_template, "smvkernel")

    if not mean:
        return _kernel(input, labels, index, index.size, ret)
    else:
        return _kernel(input, labels, index, index.size, ret, count)


def variance(input, labels=None, index=None):
    """Calculate the variance of the values of an n-D image array, optionally \
    at specified sub-regions.

    Args:
        input (cupy.ndarray) :Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output.  If None
            (default), all values where `labels` is non-zero are used.

    Returns:
       variance (cupy.ndarray): Values of variance, for each sub-region if \
            `labels` and `index` are specified.


    .. seealso:: :func:`scipy.ndimage.variance`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')

    if labels is None:
        return input.var()

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    if index is None:
        index = cupy.unique(labels)

    input, labels = cupy.broadcast_arrays(input, labels)

    if not isinstance(index, cupy.ndarray):
        raise TypeError('index must be cupy.ndarray')

    sum, count = _stats(input, labels, index, mean=True)
    mean = sum/count
    ret = cupy.zeros_like(index).astype(mean.dtype)
    return _variance_kernel()(input, labels, index,
                              index.size, mean, ret)/count


def sum(input, labels=None, index=None):
    """Calculate the sum of the values of an n-D image array, optionally \
       at specified sub-regions.

    Args:
        input (cupy.ndarray) :Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output.  If None
            (default), all values where `labels` is non-zero are used.

    Returns:
       sum (cupy.ndarray): sum of values, for each sub-region if \
            `labels` and `index` are specified.


    .. seealso:: :func:`scipy.ndimage.sum`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')

    if labels is None:
        return input.sum()

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    if index is None:
        index = cupy.unique(labels)

    input, labels = cupy.broadcast_arrays(input, labels)

    if not isinstance(index, cupy.ndarray):
        raise TypeError('index must be cupy.ndarray')

    return _stats(input, labels, index, False)


def mean(input, labels=None, index=None):
    """Calculate the mean of the values of an n-D image array, optionally \
       at specified sub-regions.

    Args:
        input (cupy.ndarray) :Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output.  If None
            (default), all values where `labels` is non-zero are used.

    Returns:
        mean (cupy.ndarray): mean of values, for each sub-region if \
            `labels` and `index` are specified.


    .. seealso:: :func:`scipy.ndimage.mean`
    """
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')

    if labels is None:
        return input.sum()/input.size

    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')

    if index is None:
        index = cupy.unique(labels)

    input, labels = cupy.broadcast_arrays(input, labels)

    if not isinstance(index, cupy.ndarray):
        raise TypeError('index must be cupy.ndarray')

    sum, count = _stats(input, labels, index, True)
    return sum/count
