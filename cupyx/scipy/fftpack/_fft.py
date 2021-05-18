from numpy import prod

import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._fft import (_convert_fft_type, _default_fft_func, _fft,
                           _get_cufft_plan_nd, _get_fftn_out_size,
                           _output_dtype)
from cupy.fft._cache import get_plan_cache


def get_fft_plan(a, shape=None, axes=None, value_type='C2C'):
    """ Generate a CUDA FFT plan for transforming up to three axes.

    Args:
        a (cupy.ndarray): Array to be transform, assumed to be either C- or
            F- contiguous.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (None or int or tuple of int):  The axes of the array to
            transform. If `None`, it is assumed that all axes are transformed.

            Currently, for performing N-D transform these must be a set of up
            to three adjacent axes, and must include either the first or the
            last axis of the array.
        value_type (str): The FFT type to perform. Acceptable values are:

            * 'C2C': complex-to-complex transform (default)
            * 'R2C': real-to-complex transform
            * 'C2R': complex-to-real transform

    Returns:
        a cuFFT plan for either 1D transform (``cupy.cuda.cufft.Plan1d``) or
        N-D transform (``cupy.cuda.cufft.PlanNd``).

    .. note::
        The returned plan can not only be passed as one of the arguments of
        the functions in ``cupyx.scipy.fftpack``, but also be used as a
        context manager for both ``cupy.fft`` and ``cupyx.scipy.fftpack``
        functions:

        .. code-block:: python

            x = cupy.random.random(16).reshape(4, 4).astype(complex)
            plan = cupyx.scipy.fftpack.get_fft_plan(x)
            with plan:
                y = cupy.fft.fftn(x)
                # alternatively:
                y = cupyx.scipy.fftpack.fftn(x)  # no explicit plan is given!
            # alternatively:
            y = cupyx.scipy.fftpack.fftn(x, plan=plan)  # pass plan explicitly

        In the first case, no cuFFT plan will be generated automatically,
        even if ``cupy.fft.config.enable_nd_planning = True`` is set.

    .. note::
        If this function is called under the context of
        :func:`~cupy.fft.config.set_cufft_callbacks`, the generated plan will
        have callbacks enabled.

    .. warning::
        This API is a deviation from SciPy's, is currently experimental, and
        may be changed in the future version.
    """
    # check input array
    if a.flags.c_contiguous:
        order = 'C'
    elif a.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('Input array a must be contiguous')

    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(axes, int):
        axes = (axes,)
    if (shape is not None) and (axes is not None) and len(shape) != len(axes):
        raise ValueError('Shape and axes have different lengths.')

    # check axes
    # n=1: 1d (need axis1D); n>1: Nd
    if axes is None:
        n = a.ndim if shape is None else len(shape)
        axes = tuple(i for i in range(-n, 0))
        if n == 1:
            axis1D = 0
    else:  # axes is a tuple
        n = len(axes)
        if n == 1:
            axis1D = axes[0]
            if axis1D >= a.ndim or axis1D < -a.ndim:
                err = 'The chosen axis ({0}) exceeds the number of '\
                      'dimensions of a ({1})'.format(axis1D, a.ndim)
                raise ValueError(err)
        elif n > 3:
            raise ValueError('Only up to three axes is supported')

    # Note that "shape" here refers to the shape along trasformed axes, not
    # the shape of the output array, and we need to convert it to the latter.
    # The result is as if "a=_cook_shape(a); return a.shape" is called.
    # Because of this, we need to use (possibly unsorted) axes.
    transformed_shape = shape
    shape = list(a.shape)
    if transformed_shape is not None:
        for s, axis in zip(transformed_shape, axes):
            if s is not None:
                if axis == axes[-1] and value_type == 'C2R':
                    s = s // 2 + 1
                shape[axis] = s
    shape = tuple(shape)

    # check value_type
    out_dtype = _output_dtype(a.dtype, value_type)
    fft_type = _convert_fft_type(out_dtype, value_type)
    # TODO(leofang): figure out if we really have to skip F-order?
    if n > 1 and value_type != 'C2C' and a.flags.f_contiguous:
        raise ValueError('C2R/R2C PlanNd for F-order arrays is not supported')

    # generate plan
    # (load from cache if it exists, otherwise create one but don't cache it)
    if n > 1:  # ND transform
        if cupy.cuda.runtime.is_hip and value_type == 'C2R':
            raise RuntimeError("hipFFT's C2R PlanNd is buggy and unsupported")
        out_size = _get_fftn_out_size(
            shape, transformed_shape, axes[-1], value_type)
        # _get_cufft_plan_nd interacts with plan cache and callback
        plan = _get_cufft_plan_nd(
            shape, fft_type, axes=axes, order=order, out_size=out_size,
            to_cache=False)
    else:  # 1D transform
        # prepare plan arguments
        if value_type != 'C2R':
            out_size = shape[axis1D]
        else:
            out_size = _get_fftn_out_size(
                shape, transformed_shape, axis1D, value_type)
        batch = prod(shape) // shape[axis1D]
        devices = None if not config.use_multi_gpus else config._devices

        keys = (out_size, fft_type, batch, devices)
        mgr = config.get_current_callback_manager()
        if mgr is not None:
            # to avoid a weird segfault, we generate and cache distinct plans
            # for every possible (load_aux, store_aux) pairs; the plans are
            # still generated from the same external Python module
            load_aux = mgr.cb_load_aux_arr
            store_aux = mgr.cb_store_aux_arr
            keys += (mgr.cb_load, mgr.cb_store,
                     0 if load_aux is None else load_aux.data.ptr,
                     0 if store_aux is None else store_aux.data.ptr)
        cache = get_plan_cache()
        cached_plan = cache.get(keys)
        if cached_plan is not None:
            plan = cached_plan
        elif mgr is None:
            plan = cufft.Plan1d(out_size, fft_type, batch, devices=devices)
        else:  # has callback
            # TODO(leofang): support multi-GPU callback (devices is ignored)
            if devices:
                raise NotImplementedError('multi-GPU cuFFT callbacks are not '
                                          'yet supported')
            plan = mgr.create_plan(('Plan1d', keys[:-3]))
            mgr.set_callbacks(plan)

    return plan


def fft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    """Compute the one-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)

            Note that `plan` is defaulted to None, meaning CuPy will use an
            auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.

    .. seealso:: :func:`scipy.fftpack.fft`
    """
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    """Compute the one-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)

            Note that `plan` is defaulted to None, meaning CuPy will use an
            auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.

    .. seealso:: :func:`scipy.fftpack.ifft`
    """
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def fft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    """Compute the two-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.fft2`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    """Compute the two-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.ifft2`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def fftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    """Compute the N-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.fftn`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


def ifftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    """Compute the N-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        shape (None or tuple of ints): Shape of the transformed axes of the
            output. If ``shape`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``shape`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fftpack.ifftn`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


def rfft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    """Compute the one-dimensional FFT for real input.

    The returned real array contains

    .. code-block:: python

        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]  # if n is even
        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]  # if n is odd

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(
                    x, axes, value_type='R2C')

            Note that `plan` is defaulted to None, meaning CuPy will either
            use an auto-generated plan behind the scene if cupy.fft.config.
            enable_nd_planning = True, or use no cuFFT plan if it is set to
            False.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fftpack.rfft`

    .. note::
       The argument `plan` is currently experimental and the interface may be
       changed in the future version.
    """
    if n is None:
        n = x.shape[axis]

    shape = list(x.shape)
    shape[axis] = n
    f = _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD, 'R2C',
             overwrite_x=overwrite_x, plan=plan)
    z = cupy.empty(shape, f.real.dtype)

    slice_z = [slice(None)] * x.ndim
    slice_f = [slice(None)] * x.ndim

    slice_z[axis] = slice(1)
    slice_f[axis] = slice(1)
    z[tuple(slice_z)] = f[tuple(slice_f)].real

    slice_z[axis] = slice(1, None, 2)
    slice_f[axis] = slice(1, None)
    z[tuple(slice_z)] = f[tuple(slice_f)].real

    slice_z[axis] = slice(2, None, 2)
    slice_f[axis] = slice(1, n - f.shape[axis] + 1)
    z[tuple(slice_z)] = f[tuple(slice_f)].imag

    return z


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """Compute the one-dimensional inverse FFT for real input.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fftpack.irfft`

    .. note::
       This function does not support a precomputed `plan`. If you need this
       capability, please consider using :func:`cupy.fft.irfft` or :func:`
       cupyx.scipy.fft.irfft`.
    """
    if n is None:
        n = x.shape[axis]
    m = min(n, x.shape[axis])

    shape = list(x.shape)
    shape[axis] = n // 2 + 1
    if x.dtype in (cupy.float16, cupy.float32):
        z = cupy.zeros(shape, dtype=cupy.complex64)
    else:
        z = cupy.zeros(shape, dtype=cupy.complex128)

    slice_x = [slice(None)] * x.ndim
    slice_z = [slice(None)] * x.ndim

    slice_x[axis] = slice(1)
    slice_z[axis] = slice(1)
    z[tuple(slice_z)].real = x[tuple(slice_x)]

    slice_x[axis] = slice(1, m, 2)
    slice_z[axis] = slice(1, m // 2 + 1)
    z[tuple(slice_z)].real = x[tuple(slice_x)]

    slice_x[axis] = slice(2, m, 2)
    slice_z[axis] = slice(1, (m + 1) // 2)
    z[tuple(slice_z)].imag = x[tuple(slice_x)]

    return _fft(z, (n,), (axis,), None, cufft.CUFFT_INVERSE, 'C2R',
                overwrite_x=overwrite_x)
