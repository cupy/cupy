from numbers import Number
import warnings

import numpy as np

import cupy

from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
                           ihfft as _ihfft, _size_last_transform_axis)
from cupy.fft import fftshift, ifftshift, fftfreq, rfftfreq

from cupyx.scipy.fftpack import get_fft_plan

__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft',
           'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq',
           'get_fft_plan']

_scipy_150 = False
try:
    import scipy
    import scipy.fft as _scipy_fft
except ImportError:
    class _DummyModule:
        def __getattr__(self, name):
            return None

    _scipy_fft = _DummyModule()
else:
    from numpy.lib import NumpyVersion as Version
    _scipy_150 = Version(scipy.__version__) >= Version('1.5.0')
    del Version
    del scipy

# Backend support for scipy.fft

__ua_domain__ = 'numpy.scipy.fft'
_implemented = {}


def __ua_convert__(dispatchables, coerce):
    if coerce:
        try:
            replaced = [
                cupy.asarray(d.value) if d.coercible and d.type is np.ndarray
                else d.value for d in dispatchables]
        except TypeError:
            return NotImplemented
    else:
        replaced = [d.value for d in dispatchables]

    if not all(d.type is not np.ndarray or isinstance(r, cupy.ndarray)
               for r, d in zip(replaced, dispatchables)):
        return NotImplemented

    return replaced


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    if 'plan' in kwargs and not _scipy_150:
        warnings.warn('The \'plan\' argument is supported in SciPy v1.5.0+')
    return fn(*args, **kwargs)


def _implements(scipy_func):
    """Decorator adds function to the dictionary of implemented functions"""
    def inner(func):
        _implemented[scipy_func] = func
        return func

    return inner


def _assequence(x):
    """Convert scalars to a sequence, otherwise pass through ``x`` unchanged"""
    if isinstance(x, Number):
        return (x,)
    return x


@_implements(_scipy_fft.fft)
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the one-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, n, axis)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.fft`
    """
    return _fft(x, (n,), (axis,), norm, cufft.CUFFT_FORWARD,
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.ifft)
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the one-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, n, axis)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.ifft`
    """
    return _fft(x, (n,), (axis,), norm, cufft.CUFFT_INVERSE,
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.fft2)
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, *, plan=None):
    """Compute the two-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.fft2`
    """
    return fftn(x, s, axes, norm, overwrite_x, plan=plan)


@_implements(_scipy_fft.ifft2)
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, *,
          plan=None):
    """Compute the two-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.ifft2`
    """
    return ifftn(x, s, axes, norm, overwrite_x, plan=plan)


@_implements(_scipy_fft.fftn)
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.fftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    func = _default_fft_func(x, s, axes)
    return func(x, s, axes, norm, cufft.CUFFT_FORWARD, overwrite_x=overwrite_x,
                plan=plan)


@_implements(_scipy_fft.ifftn)
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional inverse FFT.

    Args:
        x (cupy.ndarray): Array to be transformed.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along
            the axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes)

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and
            type will convert to complex if that of the input is another.

    .. seealso:: :func:`scipy.fft.ifftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    func = _default_fft_func(x, s, axes)
    return func(x, s, axes, norm, cufft.CUFFT_INVERSE, overwrite_x=overwrite_x,
                plan=plan)


@_implements(_scipy_fft.rfft)
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the one-dimensional FFT for real input.

    The returned array contains the positive frequency components of the
    corresponding :func:`fft`, up to and including the Nyquist frequency.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, n, axis,
                                                        value_type='R2C')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fft.rfft`

    """
    return _fft(x, (n,), (axis,), norm, cufft.CUFFT_FORWARD, 'R2C',
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.irfft)
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the one-dimensional inverse FFT for real input.

    Args:
        x (cupy.ndarray): Array to be transformed.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axis``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, n, axis,
                                                        value_type='C2R')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array.

    .. seealso:: :func:`scipy.fft.irfft`
    """
    return _fft(x, (n,), (axis,), norm, cufft.CUFFT_INVERSE, 'C2R',
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.rfft2)
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, *,
          plan=None):
    """Compute the two-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='R2C')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`scipy.fft.rfft2`
    """
    return rfftn(x, s, axes, norm, overwrite_x, plan=plan)


@_implements(_scipy_fft.irfft2)
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, *,
           plan=None):
    """Compute the two-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='C2R')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            `2*(m-1)` where `m` is the length of the final transformed axis of
            the input.

    .. seealso:: :func:`scipy.fft.irfft2`
    """
    return irfftn(x, s, axes, norm, overwrite_x, plan=plan)


@_implements(_scipy_fft.rfftn)
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='R2C')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`scipy.fft.rfftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    func = _default_fft_func(x, s, axes, value_type='R2C')
    return func(x, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C',
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.irfftn)
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, *, plan=None):
    """Compute the N-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for
            transforming ``x`` over ``axes``, which can be obtained using::

                plan = cupyx.scipy.fftpack.get_fft_plan(x, s, axes,
                                                        value_type='C2R')

            Note that ``plan`` is defaulted to ``None``, meaning CuPy will use
            an auto-generated plan behind the scene.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            ``2*(m-1)`` where `m` is the length of the final transformed axis
            of the input.

    .. seealso:: :func:`scipy.fft.irfftn`
    """
    s = _assequence(s)
    axes = _assequence(axes)
    if (10020 >= cupy.cuda.runtime.runtimeGetVersion() >= 10010
            and int(cupy.cuda.device.get_compute_capability()) < 70
            and _size_last_transform_axis(x.shape, s, axes) == 2):
        warnings.warn('Output of irfftn might not be correct due to issue '
                      'of cuFFT in CUDA 10.1/10.2 on Pascal or older GPUs.')
    func = _default_fft_func(x, s, axes, value_type='C2R')
    return func(x, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R',
                overwrite_x=overwrite_x, plan=plan)


@_implements(_scipy_fft.hfft)
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. For
            ``n`` output points, ``n//2+1`` input points are necessary. If
            ``n`` is not given, it is determined from the length of the input
            along the axis specified by ``axis``.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (None): This argument is currently not supported.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. If ``n`` is not
            given, the length of the transformed axis is ``2*(m-1)`` where `m`
            is the length of the transformed axis of the input.

    .. seealso:: :func:`scipy.fft.hfft`
    """
    # TODO(leofang): support R2C & C2R plans
    if plan is not None:
        raise NotImplementedError('hfft plan is currently not yet supported')
    return _hfft(x, n, axis, norm)


@_implements(_scipy_fft.ihfft)
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, *, plan=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Number of points along transformation axis in the
            input to use. If ``n`` is not given, the length of the input along
            the axis specified by ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (``"backward"``, ``"ortho"``, or ``"forward"``): Optional keyword
            to specify the normalization mode. Default is ``None``, which is
            an alias of ``"backward"``.
        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.
        plan (None): This argument is currently not supported.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. The length of the
            transformed axis is ``n//2+1``.

    .. seealso:: :func:`scipy.fft.ihfft`
    """
    # TODO(leofang): support R2C & C2R plans
    if plan is not None:
        raise NotImplementedError('ihfft plan is currently not yet supported')
    return _ihfft(x, n, axis, norm)
