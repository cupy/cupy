import itertools
import math
import warnings

import cupy


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise ValueError('output shape is not correct')
    else:
        dtype = output
        if dtype is None:
            dtype = input.dtype
        output = cupy.zeros(shape, dtype)
    return output


def _check_parameter(func_name, order, mode):
    if order is None:
        warnings.warn('In the current feature the default order of {} is 1. '
                      'It is different from scipy.ndimage and can change in '
                      'the future.'.format(func_name))
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')
    elif 1 < order:
        # SciPy supports order 0-5, but CuPy supports only order 0 and 1. Other
        # orders will be implemented, therefore it raises NotImplementedError
        # instead of ValueError.
        raise NotImplementedError('spline order is not supported')

    if mode in ('reflect', 'wrap'):
        raise NotImplementedError('\'{}\' mode is not supported. See '
                                  'https://github.com/scipy/scipy/issues/8465'
                                  .format(mode))
    elif mode not in ('constant', 'nearest', 'mirror', 'opencv',
                      '_opencv_edge'):
        raise ValueError('boundary mode is not supported')


def map_coordinates(input, coordinates, output=None, order=None,
                    mode='constant', cval=0.0, prefilter=True):
    """Map the input array to new coordinates by interpolation.

    The array of coordinates is used to find, for each point in the output, the
    corresponding coordinates in the input. The value of the input at those
    coordinates is determined by spline interpolation of the requested order.

    The shape of the output is derived from that of the coordinate array by
    dropping the first axis. The values of the array along the first axis are
    the coordinates in the input array at which the output value is found.

    Args:
        input (cupy.ndarray): The input array.
        coordinates (array_like): The coordinates at which ``input`` is
            evaluated.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.

    Returns:
        cupy.ndarray:
            The result of transforming the input. The shape of the output is
            derived from that of ``coordinates`` by dropping the first axis.

    .. seealso:: :func:`scipy.ndimage.map_coordinates`
    """

    _check_parameter('map_coordinates', order, mode)

    if mode == 'opencv' or mode == '_opencv_edge':
        input = cupy.pad(input, [(1, 1)] * input.ndim, 'constant',
                         constant_values=cval)
        coordinates = cupy.add(coordinates, 1)
        mode = 'constant'

    ret = _get_output(output, input, coordinates.shape[1:])

    if mode == 'nearest':
        for i in range(input.ndim):
            coordinates[i] = coordinates[i].clip(0, input.shape[i] - 1)
    elif mode == 'mirror':
        for i in range(input.ndim):
            length = input.shape[i] - 1
            if length == 0:
                coordinates[i] = 0
            else:
                coordinates[i] = cupy.remainder(coordinates[i], 2 * length)
                coordinates[i] = 2 * cupy.minimum(
                    coordinates[i], length) - coordinates[i]

    if input.dtype.kind in 'iu':
        input = input.astype(cupy.float32)

    if order == 0:
        out = input[tuple(cupy.rint(coordinates).astype(cupy.int32))]
    else:
        coordinates_floor = cupy.floor(coordinates).astype(cupy.int32)
        coordinates_ceil = coordinates_floor + 1

        sides = []
        for i in range(input.ndim):
            # TODO(mizuno): Use array_equal after it is implemented
            if cupy.all(coordinates[i] == coordinates_floor[i]):
                sides.append([0])
            else:
                sides.append([0, 1])

        out = cupy.zeros(coordinates.shape[1:], dtype=input.dtype)
        if input.dtype in (cupy.float64, cupy.complex128):
            weight = cupy.empty(coordinates.shape[1:], dtype=cupy.float64)
        else:
            weight = cupy.empty(coordinates.shape[1:], dtype=cupy.float32)
        for side in itertools.product(*sides):
            weight.fill(1)
            ind = []
            for i in range(input.ndim):
                if side[i] == 0:
                    ind.append(coordinates_floor[i])
                    weight *= coordinates_ceil[i] - coordinates[i]
                else:
                    ind.append(coordinates_ceil[i])
                    weight *= coordinates[i] - coordinates_floor[i]
            out += input[ind] * weight
        del weight

    if mode == 'constant':
        mask = cupy.zeros(coordinates.shape[1:], dtype=cupy.bool_)
        for i in range(input.ndim):
            mask += coordinates[i] < 0
            mask += coordinates[i] > input.shape[i] - 1
        out[mask] = cval
        del mask

    if ret.dtype.kind in 'iu':
        out = cupy.rint(out)
    ret[:] = out
    return ret


def affine_transform(input, matrix, offset=0.0, output_shape=None, output=None,
                     order=None, mode='constant', cval=0.0, prefilter=True):
    """Apply an affine transformation.

    Given an output image pixel index vector ``o``, the pixel value is
    determined from the input image at position
    ``cupy.dot(matrix, o) + offset``.

    Args:
        input (cupy.ndarray): The input array.
        matrix (cupy.ndarray): The inverse coordinate transformation matrix,
            mapping output coordinates to input coordinates. If ``ndim`` is the
            number of dimensions of ``input``, the given matrix must have one
            of the following shapes:

                - ``(ndim, ndim)``: the linear transformation matrix for each
                  output coordinate.
                - ``(ndim,)``: assume that the 2D transformation matrix is
                  diagonal, with the diagonal specified by the given value.
                - ``(ndim + 1, ndim + 1)``: assume that the transformation is
                  specified using homogeneous coordinates. In this case, any
                  value passed to ``offset`` is ignored.
                - ``(ndim, ndim + 1)``: as above, but the bottom row of a
                  homogeneous transformation matrix is always
                  ``[0, 0, ..., 1]``, and may be omitted.

        offset (float or sequence): The offset into the array where the
            transform is applied. If a float, ``offset`` is the same for each
            axis. If a sequence, ``offset`` should contain one value for each
            axis.
        output_shape (tuple of ints): Shape tuple.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.

    Returns:
        cupy.ndarray or None:
            The transformed input. If ``output`` is given as a parameter,
            ``None`` is returned.

    .. seealso:: :func:`scipy.ndimage.affine_transform`
    """

    _check_parameter('affine_transform', order, mode)

    if not hasattr(offset, '__iter__') and type(offset) is not cupy.ndarray:
        offset = [offset] * input.ndim

    if matrix.ndim == 1:
        # TODO(mizuno): Implement zoom_shift
        matrix = cupy.diag(matrix)
    elif matrix.shape[0] == matrix.shape[1] - 1:
        offset = matrix[:, -1]
        matrix = matrix[:, :-1]
    elif matrix.shape[0] == input.ndim + 1:
        offset = matrix[:-1, -1]
        matrix = matrix[:-1, :-1]

    if mode == 'opencv':
        m = cupy.zeros((input.ndim + 1, input.ndim + 1))
        m[:-1, :-1] = matrix
        m[:-1, -1] = offset
        m[-1, -1] = 1
        m = cupy.linalg.inv(m)
        m[:2] = cupy.roll(m[:2], 1, axis=0)
        m[:2, :2] = cupy.roll(m[:2, :2], 1, axis=1)
        matrix = m[:-1, :-1]
        offset = m[:-1, -1]

    if output_shape is None:
        output_shape = input.shape

    coordinates = cupy.indices(output_shape, dtype=cupy.float64)
    coordinates = cupy.dot(matrix, coordinates.reshape((input.ndim, -1)))
    coordinates += cupy.expand_dims(cupy.asarray(offset), -1)
    ret = _get_output(output, input, output_shape)
    ret[:] = map_coordinates(input, coordinates, ret.dtype, order, mode, cval,
                             prefilter).reshape(output_shape)
    return ret


def _minmax(coor, minc, maxc):
    if coor[0] < minc[0]:
        minc[0] = coor[0]
    if coor[0] > maxc[0]:
        maxc[0] = coor[0]
    if coor[1] < minc[1]:
        minc[1] = coor[1]
    if coor[1] > maxc[1]:
        maxc[1] = coor[1]
    return minc, maxc


def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=None,
           mode='constant', cval=0.0, prefilter=True):
    """Rotate an array.

    The array is rotated in the plane defined by the two axes given by the
    ``axes`` parameter using spline interpolation of the requested order.

    Args:
        input (cupy.ndarray): The input array.
        angle (float): The rotation angle in degrees.
        axes (tuple of 2 ints): The two axes that define the plane of rotation.
            Default is the first two axes.
        reshape (bool): If ``reshape`` is True, the output shape is adapted so
            that the input array is contained completely in the output. Default
            is True.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.

    Returns:
        cupy.ndarray or None:
            The rotated input.

    .. seealso:: :func:`scipy.ndimage.rotate`
    """

    _check_parameter('rotate', order, mode)

    if mode == 'opencv':
        mode = '_opencv_edge'

    axes = list(axes)
    if axes[0] < 0:
        axes[0] += input.ndim
    if axes[1] < 0:
        axes[1] += input.ndim
    if axes[0] > axes[1]:
        axes = axes[1], axes[0]
    if axes[0] < 0 or input.ndim <= axes[1]:
        raise IndexError

    rad = cupy.deg2rad(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)

    matrix = cupy.identity(input.ndim)
    matrix[axes[0], axes[0]] = cos
    matrix[axes[0], axes[1]] = sin
    matrix[axes[1], axes[0]] = -sin
    matrix[axes[1], axes[1]] = cos

    iy = input.shape[axes[0]]
    ix = input.shape[axes[1]]
    if reshape:
        mtrx = cupy.array([[cos, sin], [-sin, cos]])
        minc = [0, 0]
        maxc = [0, 0]
        coor = cupy.dot(mtrx, cupy.array([0, ix]))
        minc, maxc = _minmax(coor, minc, maxc)
        coor = cupy.dot(mtrx, cupy.array([iy, 0]))
        minc, maxc = _minmax(coor, minc, maxc)
        coor = cupy.dot(mtrx, cupy.array([iy, ix]))
        minc, maxc = _minmax(coor, minc, maxc)
        oy = int(maxc[0] - minc[0] + 0.5)
        ox = int(maxc[1] - minc[1] + 0.5)
    else:
        oy = input.shape[axes[0]]
        ox = input.shape[axes[1]]

    offset = cupy.zeros(input.ndim)
    offset[axes[0]] = oy / 2.0 - 0.5
    offset[axes[1]] = ox / 2.0 - 0.5
    offset = cupy.dot(matrix, offset)
    tmp = cupy.zeros(input.ndim)
    tmp[axes[0]] = iy / 2.0 - 0.5
    tmp[axes[1]] = ix / 2.0 - 0.5
    offset = tmp - offset

    output_shape = list(input.shape)
    output_shape[axes[0]] = oy
    output_shape[axes[1]] = ox

    return affine_transform(input, matrix, offset, output_shape, output, order,
                            mode, cval, prefilter)


def shift(input, shift, output=None, order=None, mode='constant', cval=0.0,
          prefilter=True):
    """Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Args:
        input (cupy.ndarray): The input array.
        shift (float or sequence): The shift along the axes. If a float,
            ``shift`` is the same for each axis. If a sequence, ``shift``
            should contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.

    Returns:
        cupy.ndarray or None:
            The shifted input.

    .. seealso:: :func:`scipy.ndimage.shift`
    """

    _check_parameter('shift', order, mode)

    if mode == 'opencv':
        mode = '_opencv_edge'

    if not hasattr(shift, '__iter__') and type(shift) is not cupy.ndarray:
        shift = [shift] * input.ndim

    return affine_transform(input, cupy.ones(input.ndim, input.dtype),
                            cupy.negative(cupy.asarray(shift)), None, output,
                            order, mode, cval, prefilter)


def zoom(input, zoom, output=None, order=None, mode='constant', cval=0.0,
         prefilter=True):
    """Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Args:
        input (cupy.ndarray): The input array.
        zoom (float or sequence): The zoom factor along the axes. If a float,
            ``zoom`` is the same for each axis. If a sequence, ``zoom`` should
            contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.

    Returns:
        cupy.ndarray or None:
            The zoomed input.

    .. seealso:: :func:`scipy.ndimage.zoom`
    """

    _check_parameter('zoom', order, mode)

    if not hasattr(zoom, '__iter__') and type(zoom) is not cupy.ndarray:
        zoom = [zoom] * input.ndim
    output_shape = []
    for s, z in zip(input.shape, zoom):
        output_shape.append(int(round(s * z)))

    if mode == 'opencv':
        zoom = []
        offset = []
        for in_size, out_size in zip(input.shape, output_shape):
            if out_size > 1:
                zoom.append(float(in_size) / out_size)
                offset.append((zoom[-1] - 1) / 2.0)
            else:
                zoom.append(0)
                offset.append(0)
        mode = 'nearest'
    else:
        zoom = []
        for in_size, out_size in zip(input.shape, output_shape):
            if out_size > 1:
                zoom.append(float(in_size - 1) / (out_size - 1))
            else:
                zoom.append(0)
        offset = 0.0

    return affine_transform(input, cupy.asarray(zoom), offset, output_shape,
                            output, order, mode, cval, prefilter)
