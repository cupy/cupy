from copy import copy
import warnings

import six

import numpy as np

import cupy
from cupy.cuda import cufft
from math import sqrt
from cupy.fft import config


def _output_dtype(a, value_type):
    if value_type != 'R2C':
        if a.dtype in [np.float16, np.float32]:
            return np.complex64
        elif a.dtype not in [np.complex64, np.complex128]:
            return np.complex128
    else:
        if a.dtype in [np.complex64, np.complex128]:
            return a.real.dtype
        elif a.dtype == np.float16:
            return np.float32
        elif a.dtype not in [np.float32, np.float64]:
            return np.float64
    return a.dtype


def _convert_dtype(a, value_type):
    out_dtype = _output_dtype(a, value_type)
    return a.astype(out_dtype, copy=False)


def _cook_shape(a, s, axes, value_type, order='C'):
    if s is None or s == a.shape:
        return a
    if (value_type == 'C2R') and (s[-1] is not None):
        s = list(s)
        s[-1] = s[-1] // 2 + 1
    for sz, axis in zip(s, axes):
        if (sz is not None) and (sz != a.shape[axis]):
            shape = list(a.shape)
            if shape[axis] > sz:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, sz)
                a = a[index]
            else:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, shape[axis])
                shape[axis] = sz
                z = cupy.zeros(shape, a.dtype.char, order=order)
                z[index] = a
                a = z
    return a


def _convert_fft_type(a, value_type):
    if value_type == 'C2C' and a.dtype == np.complex64:
        return cufft.CUFFT_C2C
    elif value_type == 'R2C' and a.dtype == np.float32:
        return cufft.CUFFT_R2C
    elif value_type == 'C2R' and a.dtype == np.complex64:
        return cufft.CUFFT_C2R
    elif value_type == 'C2C' and a.dtype == np.complex128:
        return cufft.CUFFT_Z2Z
    elif value_type == 'R2C' and a.dtype == np.float64:
        return cufft.CUFFT_D2Z
    else:
        return cufft.CUFFT_Z2D


def _exec_fft(a, direction, value_type, norm, axis, overwrite_x,
              out_size=None, out=None, plan=None):
    fft_type = _convert_fft_type(a, value_type)

    if axis % a.ndim != a.ndim - 1:
        a = a.swapaxes(axis, -1)

    if a.base is not None or not a.flags.c_contiguous:
        a = a.copy()

    if out_size is None:
        out_size = a.shape[-1]

    batch = a.size // a.shape[-1]
    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        if plan is None:
            plan = curr_plan
        else:
            raise RuntimeError('Use the cuFFT plan either as a context manager'
                               ' or as an argument.')
    if plan is None:
        plan = cufft.Plan1d(out_size, fft_type, batch)
    else:
        # check plan validity
        if not isinstance(plan, cufft.Plan1d):
            raise ValueError('expected plan to have type cufft.Plan1d')
        if fft_type != plan.fft_type:
            raise ValueError('CUFFT plan dtype mismatch.')
        if out_size != plan.nx:
            raise ValueError('Target array size does not match the plan.')
        if batch != plan.batch:
            raise ValueError('Batch size does not match the plan.')

    if overwrite_x and value_type == 'C2C':
        out = a
    elif out is not None:
        # verify that out has the expected shape and dtype
        plan.check_output_array(a, out)
    else:
        out = plan.get_output_array(a)

    plan.fft(a, out, direction)

    sz = out.shape[-1]
    if fft_type == cufft.CUFFT_R2C or fft_type == cufft.CUFFT_D2Z:
        sz = a.shape[-1]
    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            out /= sz
    else:
        out /= sqrt(sz)

    if axis % a.ndim != a.ndim - 1:
        out = out.swapaxes(axis, -1)

    return out


def _fft_c2c(a, direction, norm, axes, overwrite_x, plan=None):
    for axis in axes:
        a = _exec_fft(a, direction, 'C2C', norm, axis, overwrite_x, plan=plan)
    return a


def _fft(a, s, axes, norm, direction, value_type='C2C', overwrite_x=False,
         plan=None):
    if norm not in (None, 'ortho'):
        raise ValueError('Invalid norm value %s, should be None or "ortho".'
                         % norm)

    if s is not None:
        for n in s:
            if (n is not None) and (n < 1):
                raise ValueError(
                    'Invalid number of FFT data points (%d) specified.' % n)

    if (s is not None) and (axes is not None) and len(s) != len(axes):
        raise ValueError('Shape and axes have different lengths.')

    a = _convert_dtype(a, value_type)
    if axes is None:
        if s is None:
            dim = a.ndim
        else:
            dim = len(s)
        axes = [i for i in six.moves.range(-dim, 0)]
    a = _cook_shape(a, s, axes, value_type)

    if value_type == 'C2C':
        a = _fft_c2c(a, direction, norm, axes, overwrite_x, plan=plan)
    elif value_type == 'R2C':
        a = _exec_fft(a, direction, value_type, norm, axes[-1], overwrite_x)
        a = _fft_c2c(a, direction, norm, axes[:-1], overwrite_x)
    else:
        a = _fft_c2c(a, direction, norm, axes[:-1], overwrite_x)
        if (s is None) or (s[-1] is None):
            out_size = a.shape[axes[-1]] * 2 - 2
        else:
            out_size = s[-1]
        a = _exec_fft(a, direction, value_type, norm, axes[-1], overwrite_x,
                      out_size)

    return a


def _get_cufft_plan_nd(shape, fft_type, axes=None, order='C'):
    """Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        shape (tuple of int): The shape of the array to transform
        fft_type ({cufft.CUFFT_C2C, cufft.CUFFT_Z2Z}): The FFT type to perform.
            Currently only complex-to-complex transforms are supported.
        axes (None or int or tuple of int):  The axes of the array to
            transform. Currently, these must be a set of up to three adjacent
            axes and must include either the first or the last axis of the
            array.  If `None`, it is assumed that all axes are transformed.
        order ({'C', 'F'}): Specify whether the data to be transformed has C or
            Fortran ordered data layout.

    Returns:
        plan (cufft.PlanNd): The CUFFT Plan. This can be used with
            cufft.fft.fftn or cufft.fft.ifftn.
    """
    ndim = len(shape)

    if fft_type not in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
        raise NotImplementedError(
            'Only cufft.CUFFT_C2C and cufft.CUFFT_Z2Z are supported.')

    if axes is None:
        # transform over all axes
        fft_axes = tuple(range(ndim))
    else:
        if np.isscalar(axes):
            axes = (axes, )
        axes = tuple(axes)

        if np.min(axes) < -ndim or np.max(axes) > ndim - 1:
            raise ValueError('The specified axes exceed the array dimensions.')

        # sort the provided axes in ascending order
        fft_axes = tuple(sorted(np.mod(axes, ndim)))

        # make sure the specified axes meet the expectations made below
        if not np.all(np.diff(fft_axes) == 1):
            raise ValueError(
                'The axes to be transformed must be contiguous and repeated '
                'axes are not allowed.')
        if (0 not in fft_axes) and ((ndim - 1) not in fft_axes):
            raise ValueError(
                'Either the first or the last axis of the array must be in '
                'axes.')

    if len(fft_axes) < 1 or len(fft_axes) > 3:
        raise ValueError(
            ('CUFFT can only transform along 1, 2 or 3 axes, but {} axes were '
             'specified.').format(len(fft_axes)))

    if order not in ['C', 'F']:
        raise ValueError('order must be \'C\' or \'F\'')

    """
    For full details on idist, istride, iembed, etc. see:
    http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout

    in 1D:
    input[b * idist + x * istride]
    output[b * odist + x * ostride]

    in 2D:
    input[b * idist + (x * inembed[1] + y) * istride]
    output[b * odist + (x * onembed[1] + y) * ostride]

    in 3D:
    input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
    output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
    """
    if fft_axes == tuple(np.arange(ndim)):
        # tranfsorm over all axes
        plan_dimensions = copy(shape)
        if order == 'F':
            plan_dimensions = plan_dimensions[::-1]
        idist = np.intp(np.prod(shape))
        odist = np.intp(np.prod(shape))
        istride = ostride = 1
        inembed = onembed = None
        nbatch = 1
    else:
        plan_dimensions = []
        for d in range(ndim):
            if d in fft_axes:
                plan_dimensions.append(shape[d])
        plan_dimensions = tuple(plan_dimensions)
        if order == 'F':
            plan_dimensions = plan_dimensions[::-1]
        inembed = tuple(np.asarray(plan_dimensions, dtype=int))
        onembed = tuple(np.asarray(plan_dimensions, dtype=int))
        if 0 not in fft_axes:
            # don't FFT along the first min_axis_fft axes
            min_axis_fft = np.min(fft_axes)
            nbatch = np.prod(shape[:min_axis_fft])
            if order == 'C':
                # C-ordered GPU array with batch along first dim
                idist = np.prod(plan_dimensions)
                odist = np.prod(plan_dimensions)
                istride = 1
                ostride = 1
            elif order == 'F':
                # F-ordered GPU array with batch along first dim
                idist = 1
                odist = 1
                istride = nbatch
                ostride = nbatch
        elif (ndim - 1) not in fft_axes:
            # don't FFT along the last axis
            num_axes_batch = ndim - len(fft_axes)
            nbatch = np.prod(shape[-num_axes_batch:])
            if order == 'C':
                # C-ordered GPU array with batch along last dim
                idist = 1
                odist = 1
                istride = nbatch
                ostride = nbatch
            elif order == 'F':
                # F-ordered GPU array with batch along last dim
                idist = np.prod(plan_dimensions)
                odist = np.prod(plan_dimensions)
                istride = 1
                ostride = 1
        else:
            raise ValueError(
                'General subsets of FFT axes not currently supported for '
                'GPU case (Can only batch FFT over the first or last '
                'spatial axes).')

    plan = cufft.PlanNd(shape=plan_dimensions,
                        istride=istride,
                        ostride=ostride,
                        inembed=inembed,
                        onembed=onembed,
                        idist=idist,
                        odist=odist,
                        fft_type=fft_type,
                        batch=nbatch)
    return plan


def _exec_fftn(a, direction, value_type, norm, axes, overwrite_x,
               plan=None, out=None):

    fft_type = _convert_fft_type(a, value_type)
    if fft_type not in [cufft.CUFFT_C2C, cufft.CUFFT_Z2Z]:
        raise NotImplementedError('Only C2C and Z2Z are supported.')

    if a.base is not None:
        a = a.copy()

    if a.flags.c_contiguous:
        order = 'C'
    elif a.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('a must be contiguous')

    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        plan = curr_plan
        # don't check repeated usage; it's done in _default_fft_func()
    if plan is None:
        # generate a plan
        plan = _get_cufft_plan_nd(a.shape, fft_type, axes=axes, order=order)
    else:
        if not isinstance(plan, cufft.PlanNd):
            raise ValueError('expected plan to have type cufft.PlanNd')
        if a.flags.c_contiguous:
            expected_shape = tuple(a.shape[ax] for ax in axes)
        else:
            # plan.shape will be reversed for Fortran-ordered inputs
            expected_shape = tuple(a.shape[ax] for ax in axes[::-1])
        if expected_shape != plan.shape:
            raise ValueError(
                'The CUFFT plan and a.shape do not match: '
                'plan.shape = {}, expected_shape={}, a.shape = {}'.format(
                    plan.shape, expected_shape, a.shape))
        if fft_type != plan.fft_type:
            raise ValueError('CUFFT plan dtype mismatch.')
        # TODO: also check the strides and axes of the plan?

    if overwrite_x and value_type == 'C2C':
        out = a
    elif out is None:
        out = plan.get_output_array(a, order=order)
    else:
        plan.check_output_array(a, out)
    plan.fft(a, out, direction)

    # normalize by the product of the shape along the transformed axes
    sz = np.prod([out.shape[ax] for ax in axes])
    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            out /= sz
    else:
        out /= sqrt(sz)

    return out


def _fftn(a, s, axes, norm, direction, value_type='C2C', order='A', plan=None,
          overwrite_x=False, out=None):
    if norm not in (None, 'ortho'):
        raise ValueError('Invalid norm value %s, should be None or "ortho".'
                         % norm)

    a = _convert_dtype(a, value_type)

    if (s is not None) and (axes is not None) and len(s) != len(axes):
        raise ValueError('Shape and axes have different lengths.')

    if axes is None:
        if s is None:
            dim = a.ndim
        else:
            dim = len(s)
        axes = [i for i in six.moves.range(-dim, 0)]
    axes = tuple(axes)

    if order == 'A':
        if a.flags.f_contiguous:
            order = 'F'
        elif a.flags.c_contiguous:
            order = 'C'
        else:
            a = cupy.ascontiguousarray(a)
            order = 'C'
    elif order not in ['C', 'F']:
        raise ValueError('Unsupported order: {}'.format(order))

    a = _cook_shape(a, s, axes, value_type, order=order)
    if order == 'C' and not a.flags.c_contiguous:
        a = cupy.ascontiguousarray(a)
    elif order == 'F' and not a.flags.f_contiguous:
        a = cupy.asfortranarray(a)

    # sort the provided axes in ascending order
    axes = tuple(sorted(np.mod(axes, a.ndim)))

    a = _exec_fftn(a, direction, value_type, norm=norm, axes=axes,
                   overwrite_x=overwrite_x, plan=plan, out=out)
    return a


def _default_plan_type(a, s=None, axes=None):
    """Determine whether to use separable 1d planning or nd planning."""
    ndim = a.ndim
    if ndim == 1 or not config.enable_nd_planning:
        return '1d'

    if axes is None:
        if s is None:
            dim = ndim
        else:
            dim = len(s)
        axes = tuple([i % ndim for i in six.moves.range(-dim, 0)])
    else:
        # sort the provided axes in ascending order
        axes = tuple(sorted([i % ndim for i in axes]))

    if len(axes) == 1:
        # use Plan1d to transform a single axis
        return '1d'
    if len(axes) > 3 or not (np.all(np.diff(sorted(axes)) == 1)):
        # PlanNd supports 1d, 2d or 3d transforms over contiguous axes
        return '1d'
    if (0 not in axes) and ((ndim - 1) not in axes):
        # PlanNd only possible if the first or last axis is in axes.
        return '1d'
    return 'nd'


def _default_fft_func(a, s=None, axes=None, plan=None):
    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        if plan is None:
            plan = curr_plan
        else:
            raise RuntimeError('Use the cuFFT plan either as a context manager'
                               ' or as an argument.')

    if isinstance(plan, cufft.PlanNd):  # a shortcut for using _fftn
        return _fftn
    elif isinstance(plan, cufft.Plan1d):  # a shortcut for using _fft
        return _fft

    plan_type = _default_plan_type(a, s, axes)
    if plan_type == 'nd':
        return _fftn
    else:
        return _fft


def fft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fft`
    """
    return _fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifft`
    """
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_INVERSE)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fft2`
    """
    func = _default_fft_func(a, s, axes)
    return func(a, s, axes, norm, cufft.CUFFT_FORWARD)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifft2`
    """
    func = _default_fft_func(a, s, axes)
    return func(a, s, axes, norm, cufft.CUFFT_INVERSE)


def fftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fftn`
    """
    func = _default_fft_func(a, s, axes)
    return func(a, s, axes, norm, cufft.CUFFT_FORWARD)


def ifftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifftn`
    """
    func = _default_fft_func(a, s, axes)
    return func(a, s, axes, norm, cufft.CUFFT_INVERSE)


def rfft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Number of points along transformation axis in the
            input to use. If ``n`` is not given, the length of the input along
            the axis specified by ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. The length of the
            transformed axis is ``n//2+1``.

    .. seealso:: :func:`numpy.fft.rfft`
    """
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_FORWARD, 'R2C')


def irfft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. For
            ``n`` output points, ``n//2+1`` input points are necessary. If
            ``n`` is not given, it is determined from the length of the input
            along the axis specified by ``axis``.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. If ``n`` is not
            given, the length of the transformed axis is`2*(m-1)` where `m`
            is the length of the transformed axis of the input.

    .. seealso:: :func:`numpy.fft.irfft`
    """
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_INVERSE, 'C2R')


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`numpy.fft.rfft2`
    """
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            `2*(m-1)` where `m` is the length of the final transformed axis of
            the input.

    .. seealso:: :func:`numpy.fft.irfft2`
    """
    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


def rfftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`numpy.fft.rfftn`
    """
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


def _size_last_transform_axis(shape, s, axes):
    if s is not None:
        if s[-1] is not None:
            return s[-1]
    elif axes is not None:
        return shape[axes[-1]]
    return shape[-1]


def irfftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            ``2*(m-1)`` where `m` is the length of the final transformed axis
            of the input.

    .. seealso:: :func:`numpy.fft.irfftn`
    """
    if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010 and
            int(cupy.cuda.device.get_compute_capability()) < 70 and
            _size_last_transform_axis(a.shape, s, axes) == 2):
        warnings.warn('Output of irfftn might not be correct due to issue '
                      'of cuFFT in CUDA 10.1/10.2 on Pascal or older GPUs.')

    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


def hfft(a, n=None, axis=-1, norm=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. For
            ``n`` output points, ``n//2+1`` input points are necessary. If
            ``n`` is not given, it is determined from the length of the input
            along the axis specified by ``axis``.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. If ``n`` is not
            given, the length of the transformed axis is ``2*(m-1)`` where `m`
            is the length of the transformed axis of the input.

    .. seealso:: :func:`numpy.fft.hfft`
    """
    a = irfft(a.conj(), n, axis)
    return a * (a.shape[axis] if norm is None else
                cupy.sqrt(a.shape[axis], dtype=a.dtype))


def ihfft(a, n=None, axis=-1, norm=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Number of points along transformation axis in the
            input to use. If ``n`` is not given, the length of the input along
            the axis specified by ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. The length of the
            transformed axis is ``n//2+1``.

    .. seealso:: :func:`numpy.fft.ihfft`
    """
    if n is None:
        n = a.shape[axis]
    return rfft(a, n, axis, norm).conj() / (n if norm is None else 1)


def fftfreq(n, d=1.0):
    """Return the FFT sample frequencies.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray: Array of length ``n`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.fftfreq`
    """
    return cupy.hstack((cupy.arange(0, (n - 1) // 2 + 1, dtype=np.float64),
                        cupy.arange(-(n // 2), 0, dtype=np.float64))) / n / d


def rfftfreq(n, d=1.0):
    """Return the FFT sample frequencies for real input.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray:
            Array of length ``n//2+1`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.rfftfreq`
    """
    return cupy.arange(0, n // 2 + 1, dtype=np.float64) / n / d


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        x (cupy.ndarray): Input array.
        axes (int or tuple of ints): Axes over which to shift. Default is
            ``None``, which shifts all axes.

    Returns:
        cupy.ndarray: The shifted array.

    .. seealso:: :func:`numpy.fft.fftshift`
    """
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, x.shape[axis] // 2, axis)
    return x


def ifftshift(x, axes=None):
    """The inverse of :meth:`fftshift`.

    Args:
        x (cupy.ndarray): Input array.
        axes (int or tuple of ints): Axes over which to shift. Default is
            ``None``, which shifts all axes.

    Returns:
        cupy.ndarray: The shifted array.

    .. seealso:: :func:`numpy.fft.ifftshift`
    """
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, -(x.shape[axis] // 2), axis)
    return x
