import math
import warnings

import cupy
import numpy

from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels

_prod = cupy.core.internal.prod


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

    ret = _util._get_output(output, input, coordinates.shape[1:])
    integer_output = ret.dtype.kind in 'iu'
    _util._check_cval(mode, cval, integer_output)

    if input.dtype.kind in 'iu':
        input = input.astype(cupy.float32)

    large_int = max(_prod(input.shape), coordinates.shape[0]) > 1 << 31
    kern = _interp_kernels._get_map_kernel(
        input.ndim, large_int, yshape=coordinates.shape, mode=mode, cval=cval,
        order=order, integer_output=integer_output)
    kern(input, coordinates, ret)
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

    offset = _util._fix_sequence_arg(offset, input.ndim, 'offset', float)

    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')
    if matrix.ndim == 2:
        if matrix.shape[0] == matrix.shape[1] - 1:
            offset = matrix[:, -1]
            matrix = matrix[:, :-1]
        elif matrix.shape[0] == input.ndim + 1:
            offset = matrix[:-1, -1]
            matrix = matrix[:-1, :-1]
        if matrix.shape != (input.ndim, input.ndim):
            raise RuntimeError("improper affine shape")

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

    if mode == 'opencv' or mode == '_opencv_edge':
        if matrix.ndim == 1:
            matrix = cupy.diag(matrix)
        coordinates = cupy.indices(output_shape, dtype=cupy.float64)
        coordinates = cupy.dot(matrix, coordinates.reshape((input.ndim, -1)))
        coordinates += cupy.expand_dims(cupy.asarray(offset), -1)
        ret = _util._get_output(output, input, shape=output_shape)
        ret[:] = map_coordinates(input, coordinates, ret.dtype, order, mode,
                                 cval, prefilter).reshape(output_shape)
        return ret

    matrix = matrix.astype(cupy.float64, copy=False)
    if order is None:
        order = 1
    ndim = input.ndim
    output = _util._get_output(output, input, shape=output_shape)
    if input.dtype.kind in 'iu':
        input = input.astype(cupy.float32)

    integer_output = output.dtype.kind in 'iu'
    _util._check_cval(mode, cval, integer_output)
    large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
    if matrix.ndim == 1:
        offset = cupy.asarray(offset, dtype=cupy.float64)
        offset = -offset / matrix
        kern = _interp_kernels._get_zoom_shift_kernel(
            ndim, large_int, output_shape, mode, cval=cval, order=order,
            integer_output=integer_output)
        kern(input, offset, matrix, output)
    else:
        kern = _interp_kernels._get_affine_kernel(
            ndim, large_int, output_shape, mode, cval=cval, order=order,
            integer_output=integer_output)
        m = cupy.zeros((ndim, ndim + 1), dtype=cupy.float64)
        m[:, :-1] = matrix
        m[:, -1] = cupy.asarray(offset, dtype=cupy.float64)
        kern(input, m, output)
    return output


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

    input_arr = input
    axes = list(axes)
    if axes[0] < 0:
        axes[0] += input_arr.ndim
    if axes[1] < 0:
        axes[1] += input_arr.ndim
    if axes[0] > axes[1]:
        axes = [axes[1], axes[0]]
    if axes[0] < 0 or input_arr.ndim <= axes[1]:
        raise ValueError('invalid rotation plane specified')

    ndim = input_arr.ndim
    rad = numpy.deg2rad(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)

    # determine offsets and output shape as in scipy.ndimage.rotate
    rot_matrix = numpy.array([[cos, sin],
                              [-sin, cos]])

    img_shape = numpy.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]
    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ [[0, 0, iy, iy],
                                   [0, ix, 0, ix]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(cupy.int64)
    else:
        out_plane_shape = img_shape[axes]

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    matrix = numpy.identity(ndim)
    matrix[axes[0], axes[0]] = cos
    matrix[axes[0], axes[1]] = sin
    matrix[axes[1], axes[0]] = -sin
    matrix[axes[1], axes[1]] = cos

    iy = input.shape[axes[0]]
    offset = numpy.zeros(ndim, dtype=cupy.float64)
    offset[axes] = in_center - out_center

    matrix = cupy.asarray(matrix)
    offset = cupy.asarray(offset)

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

    shift = _util._fix_sequence_arg(shift, input.ndim, 'shift', float)

    if mode == 'opencv':
        mode = '_opencv_edge'

        output = affine_transform(
            input,
            cupy.ones(input.ndim, input.dtype),
            cupy.negative(cupy.asarray(shift)),
            None,
            output,
            order,
            mode,
            cval,
            prefilter,
        )
    else:
        if order is None:
            order = 1
        output = _util._get_output(output, input)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = _prod(input.shape) > 1 << 31
        kern = _interp_kernels._get_shift_kernel(
            input.ndim, large_int, input.shape, mode, cval=cval, order=order,
            integer_output=integer_output)
        shift = cupy.asarray(shift, dtype=cupy.float64)
        kern(input, shift, output)
    return output


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

    zoom = _util._fix_sequence_arg(zoom, input.ndim, 'zoom', float)

    output_shape = []
    for s, z in zip(input.shape, zoom):
        output_shape.append(int(round(s * z)))
    output_shape = tuple(output_shape)

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

        output = affine_transform(
            input,
            cupy.asarray(zoom),
            offset,
            output_shape,
            output,
            order,
            mode,
            cval,
            prefilter,
        )
    else:
        if order is None:
            order = 1

        zoom = []
        for in_size, out_size in zip(input.shape, output_shape):
            if out_size > 1:
                zoom.append(float(in_size - 1) / (out_size - 1))
            else:
                zoom.append(0)

        output = _util._get_output(output, input, shape=output_shape)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
        kern = _interp_kernels._get_zoom_kernel(
            input.ndim, large_int, output_shape, mode, order=order,
            integer_output=integer_output)
        zoom = cupy.asarray(zoom, dtype=cupy.float64)
        kern(input, zoom, output)
    return output
