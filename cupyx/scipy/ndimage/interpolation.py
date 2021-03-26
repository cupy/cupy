import math
import warnings

import cupy
import numpy

from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core

_prod = cupy._core.internal.prod


def _check_parameter(func_name, order, mode):
    if order is None:
        warnings.warn(f'Currently the default order of {func_name} is 1. In a '
                      'future release this may change to 3 to match '
                      'scipy.ndimage ')
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')

    if mode not in ('constant', 'grid-constant', 'nearest', 'mirror',
                    'reflect', 'grid-mirror', 'wrap', 'grid-wrap', 'opencv',
                    '_opencv_edge'):
        raise ValueError('boundary mode ({}) is not supported'.format(mode))


def _get_spline_output(input, output):
    """Create workspace array, temp, and the final dtype for the output.

    Differs from SciPy by not always forcing the internal floating point dtype
    to be double precision.
    """
    complex_data = input.dtype.kind == 'c'
    if complex_data:
        min_float_dtype = cupy.complex64
    else:
        min_float_dtype = cupy.float32
    if isinstance(output, cupy.ndarray):
        if complex_data and output.dtype.kind != 'c':
            raise ValueError(
                'output must have complex dtype for complex inputs'
            )
        float_dtype = cupy.promote_types(output.dtype, min_float_dtype)
        output_dtype = output.dtype
    else:
        if output is None:
            output = output_dtype = input.dtype
        else:
            output_dtype = cupy.dtype(output)
        float_dtype = cupy.promote_types(output, min_float_dtype)

    if (isinstance(output, cupy.ndarray)
            and output.dtype == float_dtype == output_dtype
            and output.flags.c_contiguous):
        if output is not input:
            output[...] = input[...]
        temp = output
    else:
        temp = input.astype(float_dtype, copy=False)
        temp = cupy.ascontiguousarray(temp)
        if cupy.shares_memory(temp, input, 'MAY_SHARE_BOUNDS'):
            temp = temp.copy()
    return temp, float_dtype, output_dtype


def spline_filter1d(input, order=3, axis=-1, output=cupy.float64,
                    mode='mirror'):
    """
    Calculate a 1-D spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Args:
        input (cupy.ndarray): The input array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        axis (int): The axis along which the spline filter is applied. Default
            is the last axis.
        output (cupy.ndarray or dtype, optional): The array in which to place
            the output, or the dtype of the returned array. Default is
            ``numpy.float64``.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).

    Returns:
        cupy.ndarray: The result of prefiltering the input.

    .. seealso:: :func:`scipy.spline_filter1d`
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    x = input
    ndim = x.ndim
    axis = internal._normalize_axis_index(axis, ndim)

    # order 0, 1 don't require reshaping as no CUDA kernel will be called
    # scalar or size 1 arrays also don't need to be filtered
    run_kernel = not (order < 2 or x.ndim == 0 or x.shape[axis] == 1)
    if not run_kernel:
        output = _util._get_output(output, input)
        output[...] = x[...]
        return output

    temp, data_dtype, output_dtype = _get_spline_output(x, output)
    data_type = cupy._core._scalar.get_typename(temp.dtype)
    pole_type = cupy._core._scalar.get_typename(temp.real.dtype)

    index_type = _util._get_inttype(input)
    index_dtype = cupy.int32 if index_type == 'int' else cupy.int64

    n_samples = x.shape[axis]
    n_signals = x.size // n_samples
    info = cupy.array((n_signals, n_samples) + x.shape, dtype=index_dtype)

    # empirical choice of block size that seemed to work well
    block_size = max(2 ** math.ceil(numpy.log2(n_samples / 32)), 8)
    kern = _spline_prefilter_core.get_raw_spline1d_kernel(
        axis,
        ndim,
        mode,
        order=order,
        index_type=index_type,
        data_type=data_type,
        pole_type=pole_type,
        block_size=block_size,
    )

    # Due to recursive nature, a given line of data must be processed by a
    # single thread. n_signals lines will be processed in total.
    block = (block_size,)
    grid = ((n_signals + block[0] - 1) // block[0],)

    # apply prefilter gain
    poles = _spline_prefilter_core.get_poles(order=order)
    temp *= _spline_prefilter_core.get_gain(poles)

    # apply caual + anti-causal IIR spline filters
    kern(grid, block, (temp, info))

    if isinstance(output, cupy.ndarray) and temp is not output:
        # copy kernel output into the user-provided output array
        output[...] = temp[...]
        return output
    return temp.astype(output_dtype, copy=False)


def spline_filter(input, order=3, output=cupy.float64, mode='mirror'):
    """Multidimensional spline filter.

    Args:
        input (cupy.ndarray): The input array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        output (cupy.ndarray or dtype, optional): The array in which to place
            the output, or the dtype of the returned array. Default is
            ``numpy.float64``.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).

    Returns:
        cupy.ndarray: The result of prefiltering the input.

    .. seealso:: :func:`scipy.spline_filter1d`
    """
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')

    x = input
    temp, data_dtype, output_dtype = _get_spline_output(x, output)
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(x.ndim):
            spline_filter1d(x, order, axis, output=temp, mode=mode)
            x = temp
    if isinstance(output, cupy.ndarray):
        output[...] = temp[...]
    else:
        output = temp
    if output.dtype != output_dtype:
        output = output.astype(output_dtype)
    return output


def _check_coordinates(coordinates, order, allow_float32=True):
    if coordinates.dtype.kind == 'f':
        if allow_float32:
            coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float32)
        else:
            coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float64)
        coordinates = coordinates.astype(coord_dtype, copy=False)
    elif coordinates.dtype.kind in 'iu':
        if order > 1:
            # order > 1 (spline) kernels require floating-point coordinates
            if allow_float32:
                coord_dtype = cupy.promote_types(
                    coordinates.dtype, cupy.float32
                )
            else:
                coord_dtype = cupy.promote_types(
                    coordinates.dtype, cupy.float64
                )
            coordinates = coordinates.astype(coord_dtype)
    else:
        raise ValueError('coordinates should have floating point dtype')
    if not coordinates.flags.c_contiguous:
        coordinates = cupy.ascontiguousarray(coordinates)
    return coordinates


def _prepad_for_spline_filter(input, mode, cval):
    if mode in ['nearest', 'grid-constant']:
        # these modes need padding to get accurate boundary values
        npad = 12  # empirical factor chosen by SciPy
        if mode == 'grid-constant':
            kwargs = dict(mode='constant', constant_values=cval)
        else:
            kwargs = dict(mode='edge')
        padded = cupy.pad(input, npad, **kwargs)
    else:
        npad = 0
        padded = input
    return padded, npad


def _filter_input(image, prefilter, mode, cval, order):
    """Perform spline prefiltering when needed.

    Spline orders > 1 need a prefiltering stage to preserve resolution.

    For boundary modes without analytical spline boundary conditions, some
    prepadding of the input with cupy.pad is used to maintain accuracy.
    ``npad`` is an integer corresponding to the amount of padding at each edge
    of the array.
    """
    if not prefilter or order < 2:
        return (cupy.ascontiguousarray(image), 0)
    padded, npad = _prepad_for_spline_filter(image, mode, cval)
    float_dtype = cupy.promote_types(image.dtype, cupy.float32)
    filtered = spline_filter(padded, order, output=float_dtype, mode=mode)
    return cupy.ascontiguousarray(filtered), npad


def map_coordinates(input, coordinates, output=None, order=3,
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
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
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
    coordinates = _check_coordinates(coordinates, order)
    filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
    large_int = max(_prod(input.shape), coordinates.shape[0]) > 1 << 31
    kern = _interp_kernels._get_map_kernel(
        input.ndim, large_int, yshape=coordinates.shape, mode=mode, cval=cval,
        order=order, integer_output=integer_output, nprepad=nprepad)
    kern(filtered, coordinates, ret)
    return ret


def affine_transform(input, matrix, offset=0.0, output_shape=None, output=None,
                     order=3, mode='constant', cval=0.0, prefilter=True):
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
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
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
            raise RuntimeError('improper affine shape')

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
    ndim = input.ndim
    output = _util._get_output(output, input, shape=output_shape)
    if input.dtype.kind in 'iu':
        input = input.astype(cupy.float32)
    filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)

    integer_output = output.dtype.kind in 'iu'
    _util._check_cval(mode, cval, integer_output)
    large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
    if matrix.ndim == 1:
        offset = cupy.asarray(offset, dtype=cupy.float64)
        offset = -offset / matrix
        kern = _interp_kernels._get_zoom_shift_kernel(
            ndim, large_int, output_shape, mode, cval=cval, order=order,
            integer_output=integer_output, nprepad=nprepad)
        kern(filtered, offset, matrix, output)
    else:
        kern = _interp_kernels._get_affine_kernel(
            ndim, large_int, output_shape, mode, cval=cval, order=order,
            integer_output=integer_output, nprepad=nprepad)
        m = cupy.zeros((ndim, ndim + 1), dtype=cupy.float64)
        m[:, :-1] = matrix
        m[:, -1] = cupy.asarray(offset, dtype=cupy.float64)
        kern(filtered, m, output)
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


def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
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
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
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

    offset = numpy.zeros(ndim, dtype=cupy.float64)
    offset[axes] = in_center - out_center

    matrix = cupy.asarray(matrix)
    offset = cupy.asarray(offset)

    return affine_transform(input, matrix, offset, output_shape, output, order,
                            mode, cval, prefilter)


def shift(input, shift, output=None, order=3, mode='constant', cval=0.0,
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
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
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
        output = _util._get_output(output, input)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = _prod(input.shape) > 1 << 31
        kern = _interp_kernels._get_shift_kernel(
            input.ndim, large_int, input.shape, mode, cval=cval, order=order,
            integer_output=integer_output, nprepad=nprepad)
        shift = cupy.asarray(shift, dtype=cupy.float64, order='C')
        if shift.ndim != 1:
            raise ValueError('shift must be 1d')
        if shift.size != filtered.ndim:
            raise ValueError('len(shift) must equal input.ndim')
        kern(filtered, shift, output)
    return output


def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True, *, grid_mode=False):
    """Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Args:
        input (cupy.ndarray): The input array.
        zoom (float or sequence): The zoom factor along the axes. If a float,
            ``zoom`` is the same for each axis. If a sequence, ``zoom`` should
            contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.
        grid_mode (bool, optional): If False, the distance from the pixel
            centers is zoomed. Otherwise, the distance including the full pixel
            extent is used. For example, a 1d signal of length 5 is considered
            to have length 4 when ``grid_mode`` is False, but length 5 when
            ``grid_mode`` is True. See the following visual illustration:

            .. code-block:: text

                    | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                         |<-------------------------------------->|
                                            vs.
                    |<----------------------------------------------->|

            The starting point of the arrow in the diagram above corresponds to
            coordinate location 0 in each mode.

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
        if grid_mode:

            # warn about modes that may have surprising behavior
            suggest_mode = None
            if mode == 'constant':
                suggest_mode = 'grid-constant'
            elif mode == 'wrap':
                suggest_mode = 'grid-wrap'
            if suggest_mode is not None:
                warnings.warn(
                    f'It is recommended to use mode = {suggest_mode} instead '
                    f'of {mode} when grid_mode is True.')

        zoom = []
        for in_size, out_size in zip(input.shape, output_shape):
            if grid_mode and out_size > 0:
                zoom.append(in_size / out_size)
            elif out_size > 1:
                zoom.append((in_size - 1) / (out_size - 1))
            else:
                zoom.append(0)

        output = _util._get_output(output, input, shape=output_shape)
        if input.dtype.kind in 'iu':
            input = input.astype(cupy.float32)
        filtered, nprepad = _filter_input(input, prefilter, mode, cval, order)
        integer_output = output.dtype.kind in 'iu'
        _util._check_cval(mode, cval, integer_output)
        large_int = max(_prod(input.shape), _prod(output_shape)) > 1 << 31
        kern = _interp_kernels._get_zoom_kernel(
            input.ndim, large_int, output_shape, mode, order=order,
            integer_output=integer_output, grid_mode=grid_mode,
            nprepad=nprepad)
        zoom = cupy.asarray(zoom, dtype=cupy.float64)
        kern(filtered, zoom, output)
    return output
