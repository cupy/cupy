import functools
import math
import warnings

import numpy as np

import cupy
from cupy.cuda import cufft
from cupy.fft import config


_reduce = functools.reduce
_prod = cupy.core.internal.prod


@cupy.util.memoize()
def _output_dtype(dtype, value_type):
    if value_type != 'R2C':
        if dtype in [np.float16, np.float32]:
            return np.complex64
        elif dtype not in [np.complex64, np.complex128]:
            return np.complex128
    else:
        if dtype in [np.complex64, np.complex128]:
            return np.dtype(dtype.char.lower())
        elif dtype == np.float16:
            return np.float32
        elif dtype not in [np.float32, np.float64]:
            return np.float64
    return dtype


def _convert_dtype(a, value_type):
    out_dtype = _output_dtype(a.dtype, value_type)
    if out_dtype != a.dtype:
        a = a.astype(out_dtype)
    return a


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


def _convert_fft_type(dtype, value_type):
    if value_type == 'C2C' and dtype == np.complex64:
        return cufft.CUFFT_C2C
    elif value_type == 'R2C' and dtype == np.float32:
        return cufft.CUFFT_R2C
    elif value_type == 'C2R' and dtype == np.complex64:
        return cufft.CUFFT_C2R
    elif value_type == 'C2C' and dtype == np.complex128:
        return cufft.CUFFT_Z2Z
    elif value_type == 'R2C' and dtype == np.float64:
        return cufft.CUFFT_D2Z
    elif value_type == 'C2R' and dtype == np.complex128:
        return cufft.CUFFT_Z2D
    else:
        raise ValueError


def _exec_fft(a, direction, value_type, norm, axis, overwrite_x,
              out_size=None, out=None, plan=None):
    fft_type = _convert_fft_type(a.dtype, value_type)

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
        devices = None if not config.use_multi_gpus else config._devices
        plan = cufft.Plan1d(out_size, fft_type, batch, devices=devices)
    else:
        # check plan validity
        if not isinstance(plan, cufft.Plan1d):
            raise ValueError('expected plan to have type cufft.Plan1d')
        if fft_type != plan.fft_type:
            raise ValueError('cuFFT plan dtype mismatch.')
        if out_size != plan.nx:
            raise ValueError('Target array size does not match the plan.',
                             out_size, plan.nx)
        if batch != plan.batch:
            raise ValueError('Batch size does not match the plan.')
        if config.use_multi_gpus != plan._use_multi_gpus:
            raise ValueError('Unclear if multiple GPUs are to be used or not.')

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
        out /= math.sqrt(sz)

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

    if axes is None:
        if s is None:
            dim = a.ndim
        else:
            dim = len(s)
        axes = [i for i in range(-dim, 0)]
    else:
        axes = tuple(axes)
    if not axes:
        if value_type == 'C2C':
            return a
        else:
            raise IndexError('list index out of range')
    a = _convert_dtype(a, value_type)
    a = _cook_shape(a, s, axes, value_type)

    if value_type == 'C2C':
        a = _fft_c2c(a, direction, norm, axes, overwrite_x, plan=plan)
    elif value_type == 'R2C':
        a = _exec_fft(a, direction, value_type, norm, axes[-1], overwrite_x)
        a = _fft_c2c(a, direction, norm, axes[:-1], overwrite_x)
    else:  # C2R
        a = _fft_c2c(a, direction, norm, axes[:-1], overwrite_x)
        # _cook_shape tells us input shape only, and no output shape
        out_size = _get_fftn_out_size(a.shape, s, axes[-1], value_type)
        a = _exec_fft(a, direction, value_type, norm, axes[-1], overwrite_x,
                      out_size)

    return a


def _prep_fftn_axes(ndim, s=None, axes=None, value_type='C2C'):
    """Configure axes argument for an n-dimensional FFT.

    The axes to be transformed are returned in ascending order.
    """

    # compatibility checks for cupy.cuda.cufft.PlanNd
    if (s is not None) and (axes is not None) and len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")

    if axes is None:
        if s is None:
            dim = ndim
        else:
            dim = len(s)
        axes = tuple([i + ndim for i in range(-dim, 0)])
        axes_sorted = axes
    else:
        axes = tuple(axes)
        if not axes:
            return (), ()
        if _reduce(min, axes) < -ndim or _reduce(max, axes) > ndim - 1:
            raise ValueError("The specified axes exceed the array dimensions.")
        if value_type == 'C2C':
            axes_sorted = tuple(sorted([ax % ndim for ax in axes]))
        else:  # C2R or R2C
            # The last axis is special, need to isolate it and append
            # to the rest of (sorted) axes
            axes_sorted = sorted([ax % ndim for ax in axes[:-1]])
            axes_sorted.append(axes[-1] % ndim)
            axes_sorted = tuple(axes_sorted)

    # unsorted axes for _cook_shape, sorted ones are otherwise used
    return axes, axes_sorted


def _nd_plan_is_possible(axes_sorted, ndim):
    # PlanNd supports 1D, 2D and 3D batch transforms over contiguous axes
    # Axes must be contiguous and the first or last axis must be in the axes.
    return (0 < len(axes_sorted) <= 3
            and (0 in axes_sorted or (ndim - 1) in axes_sorted)
            and all([
                (axes_sorted[n + 1] - axes_sorted[n]) == 1
                for n in range(len(axes_sorted) - 1)]))


def _get_cufft_plan_nd(shape, fft_type, axes=None, order='C', out_size=None):
    """Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        shape (tuple of int): The shape of the array to transform
        fft_type (int): The FFT type to perform. Supported values are:
            `cufft.CUFFT_C2C`, `cufft.CUFFT_C2R`, `cufft.CUFFT_R2C`,
            `cufft.CUFFT_Z2Z`, `cufft.CUFFT_Z2D`, and `cufft.CUFFT_D2Z`.
        axes (None or int or tuple of int):  The axes of the array to
            transform. Currently, these must be a set of up to three adjacent
            axes and must include either the first or the last axis of the
            array.  If `None`, it is assumed that all axes are transformed.
        order ({'C', 'F'}): Specify whether the data to be transformed has C or
            Fortran ordered data layout.
        out_size (int): The output length along the last axis for R2C/C2R FFTs.
            For C2C FFT, this is ignored (and set to `None`).

    Returns:
        plan (cufft.PlanNd): A cuFFT Plan for the chosen `fft_type`.
    """
    ndim = len(shape)

    if fft_type in (cufft.CUFFT_C2C, cufft.CUFFT_Z2Z):
        value_type = 'C2C'
    elif fft_type in (cufft.CUFFT_C2R, cufft.CUFFT_Z2D):
        value_type = 'C2R'
    else:  # CUFFT_R2C or CUFFT_D2Z
        value_type = 'R2C'

    if axes is None:
        # transform over all axes
        fft_axes = tuple(range(ndim))
    else:
        _, fft_axes = _prep_fftn_axes(ndim, s=None, axes=axes,
                                      value_type=value_type)

    if not _nd_plan_is_possible(fft_axes, ndim):
        raise ValueError(
            "An n-dimensional cuFFT plan could not be created. The axes must "
            "be contiguous and non-repeating. Between one and three axes can "
            "be transformed and either the first or last axis must be "
            "included in axes.")

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
    # At this point, _default_fft_func() guarantees that for F-order arrays
    # we only need to consider C2C, and not C2R or R2C.
    # TODO(leofang): figure out if we really have to skip F-order?
    in_dimensions = [shape[d] for d in fft_axes]
    if order == 'F':
        in_dimensions = in_dimensions[::-1]
    in_dimensions = tuple(in_dimensions)
    if fft_type in (cufft.CUFFT_C2C, cufft.CUFFT_Z2Z):
        out_dimensions = in_dimensions
        plan_dimensions = in_dimensions
    else:
        out_dimensions = list(in_dimensions)
        if out_size is not None:  # for C2R & R2C
            out_dimensions[-1] = out_size  # only valid for C order!
        out_dimensions = tuple(out_dimensions)
        if fft_type in (cufft.CUFFT_R2C, cufft.CUFFT_D2Z):
            plan_dimensions = in_dimensions
        else:  # CUFFT_C2R or CUFFT_Z2D
            plan_dimensions = out_dimensions
    inembed = in_dimensions
    onembed = out_dimensions

    if fft_axes == tuple(range(ndim)):
        # tranfsorm over all axes
        nbatch = 1
        idist = odist = 1  # doesn't matter since nbatch = 1
        istride = ostride = 1
    else:
        # batch along the first or the last axis
        if 0 not in fft_axes:
            # don't FFT along the first min_axis_fft axes
            min_axis_fft = _reduce(min, fft_axes)
            nbatch = _prod(shape[:min_axis_fft])
            if order == 'C':
                # C-ordered GPU array with batch along first dim
                idist = _prod(in_dimensions)
                odist = _prod(out_dimensions)
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
            nbatch = _prod(shape[-num_axes_batch:])
            if order == 'C':
                # C-ordered GPU array with batch along last dim
                idist = 1
                odist = 1
                istride = nbatch
                ostride = nbatch
            elif order == 'F':
                # F-ordered GPU array with batch along last dim
                idist = _prod(in_dimensions)
                odist = _prod(out_dimensions)
                istride = 1
                ostride = 1
        else:
            raise ValueError(
                'General subsets of FFT axes not currently supported for '
                'GPU case (Can only batch FFT over the first or last '
                'spatial axes).')

    plan = cufft.PlanNd(shape=plan_dimensions,
                        inembed=inembed,
                        istride=istride,
                        idist=idist,
                        onembed=onembed,
                        ostride=ostride,
                        odist=odist,
                        fft_type=fft_type,
                        batch=nbatch,
                        order=order,
                        last_axis=fft_axes[-1],
                        last_size=out_size)
    return plan


def _get_fftn_out_size(in_shape, s, last_axis, value_type):
    if value_type == 'C2R':
        if (s is None) or (s[-1] is None):
            out_size = 2 * (in_shape[last_axis] - 1)
        else:
            out_size = s[-1]
    elif value_type == 'R2C':
        out_size = in_shape[last_axis] // 2 + 1
    else:  # C2C
        out_size = None
    return out_size


def _exec_fftn(a, direction, value_type, norm, axes, overwrite_x,
               plan=None, out=None, out_size=None):

    fft_type = _convert_fft_type(a.dtype, value_type)

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
        plan = _get_cufft_plan_nd(a.shape, fft_type, axes=axes, order=order,
                                  out_size=out_size)
    else:
        if not isinstance(plan, cufft.PlanNd):
            raise ValueError('expected plan to have type cufft.PlanNd')
        if order != plan.order:
            raise ValueError('array orders mismatch (plan: {}, input: {})'
                             .format(plan.order, order))
        if a.flags.c_contiguous:
            expected_shape = [a.shape[ax] for ax in axes]
            if value_type == 'C2R':
                expected_shape[-1] = out_size
        else:
            # plan.shape will be reversed for Fortran-ordered inputs
            expected_shape = [a.shape[ax] for ax in axes[::-1]]
            # TODO(leofang): modify the shape for C2R
        expected_shape = tuple(expected_shape)
        if expected_shape != plan.shape:
            raise ValueError(
                'The cuFFT plan and a.shape do not match: '
                'plan.shape = {}, expected_shape={}, a.shape = {}'.format(
                    plan.shape, expected_shape, a.shape))
        if fft_type != plan.fft_type:
            raise ValueError('cuFFT plan dtype mismatch.')
        if value_type != 'C2C':
            if axes[-1] != plan.last_axis:
                raise ValueError('The last axis for R2C/C2R mismatch')
            if out_size != plan.last_size:
                raise ValueError('The size along the last R2C/C2R axis '
                                 'mismatch')

    # TODO(leofang): support in-place transform for R2C/C2R
    if overwrite_x and value_type == 'C2C':
        out = a
    elif out is None:
        out = plan.get_output_array(a, order=order)
    else:
        plan.check_output_array(a, out)
    plan.fft(a, out, direction)

    # normalize by the product of the shape along the transformed axes
    arr = a if fft_type in (cufft.CUFFT_R2C, cufft.CUFFT_D2Z) else out
    sz = _prod([arr.shape[ax] for ax in axes])
    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            out /= sz
    else:
        out /= math.sqrt(sz)

    return out


def _fftn(a, s, axes, norm, direction, value_type='C2C', order='A', plan=None,
          overwrite_x=False, out=None):
    if norm not in (None, 'ortho'):
        raise ValueError('Invalid norm value %s, should be None or "ortho".'
                         % norm)

    axes, axes_sorted = _prep_fftn_axes(a.ndim, s, axes, value_type)
    if not axes_sorted:
        if value_type == 'C2C':
            return a
        else:
            raise IndexError('list index out of range')
    a = _convert_dtype(a, value_type)

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

    # Note: need to call _cook_shape prior to sorting the axes
    a = _cook_shape(a, s, axes, value_type, order=order)

    if order == 'C' and not a.flags.c_contiguous:
        a = cupy.ascontiguousarray(a)
    elif order == 'F' and not a.flags.f_contiguous:
        a = cupy.asfortranarray(a)

    # _cook_shape tells us input shape only, and not output shape
    out_size = _get_fftn_out_size(a.shape, s, axes_sorted[-1], value_type)

    a = _exec_fftn(a, direction, value_type, norm=norm, axes=axes_sorted,
                   overwrite_x=overwrite_x, plan=plan, out=out,
                   out_size=out_size)
    return a


def _default_fft_func(a, s=None, axes=None, plan=None, value_type='C2C'):
    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        if plan is None:
            plan = curr_plan
        else:
            raise RuntimeError('Use the cuFFT plan either as a context manager'
                               ' or as an argument.')

    if isinstance(plan, cufft.PlanNd):  # a shortcut for using _fftn
        return _fftn
    elif (isinstance(plan, cufft.Plan1d) or
          a.ndim == 1 or not config.enable_nd_planning):
        return _fft

    # cuFFT's N-D C2R/R2C transforms may not agree with NumPy's outcomes
    if a.flags.f_contiguous and value_type != 'C2C':
        return _fft

    _, axes_sorted = _prep_fftn_axes(a.ndim, s, axes, value_type)
    if len(axes_sorted) > 1 and _nd_plan_is_possible(axes_sorted, a.ndim):
        # prefer Plan1D in the 1D case
        return _fftn
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
    func = _default_fft_func(a, s, axes, value_type='R2C')
    return func(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


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
    func = _default_fft_func(a, s, axes, value_type='C2R')
    return func(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


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
    func = _default_fft_func(a, s, axes, value_type='R2C')
    return func(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


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
    if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010
            and int(cupy.cuda.device.get_compute_capability()) < 70
            and _size_last_transform_axis(a.shape, s, axes) == 2):
        warnings.warn('Output of irfftn might not be correct due to issue '
                      'of cuFFT in CUDA 10.1/10.2 on Pascal or older GPUs.')

    func = _default_fft_func(a, s, axes, value_type='C2R')
    return func(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


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
        axes = list(range(x.ndim))
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
        axes = list(range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, -(x.shape[axis] // 2), axis)
    return x
