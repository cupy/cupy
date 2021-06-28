import numpy

import cupy
from cupy import _core
from cupy._math import sumprod
from cupy._math import ufunc


sin = ufunc.create_math_ufunc(
    'sin', 1, 'cupy_sin',
    '''Elementwise sine function.

    .. seealso:: :data:`numpy.sin`

    ''')


cos = ufunc.create_math_ufunc(
    'cos', 1, 'cupy_cos',
    '''Elementwise cosine function.

    .. seealso:: :data:`numpy.cos`

    ''')


tan = ufunc.create_math_ufunc(
    'tan', 1, 'cupy_tan',
    '''Elementwise tangent function.

    .. seealso:: :data:`numpy.tan`

    ''')


arcsin = ufunc.create_math_ufunc(
    'asin', 1, 'cupy_arcsin',
    '''Elementwise inverse-sine function (a.k.a. arcsine function).

    .. seealso:: :data:`numpy.arcsin`

    ''')


arccos = ufunc.create_math_ufunc(
    'acos', 1, 'cupy_arccos',
    '''Elementwise inverse-cosine function (a.k.a. arccosine function).

    .. seealso:: :data:`numpy.arccos`

    ''')


arctan = ufunc.create_math_ufunc(
    'atan', 1, 'cupy_arctan',
    '''Elementwise inverse-tangent function (a.k.a. arctangent function).

    .. seealso:: :data:`numpy.arctan`

    ''')


hypot = ufunc.create_math_ufunc(
    'hypot', 2, 'cupy_hypot',
    '''Computes the hypoteneous of orthogonal vectors of given length.

    This is equivalent to ``sqrt(x1 **2 + x2 ** 2)``, while this function is
    more efficient.

    .. seealso:: :data:`numpy.hypot`

    ''')


arctan2 = ufunc.create_math_ufunc(
    'atan2', 2, 'cupy_arctan2',
    '''Elementwise inverse-tangent of the ratio of two arrays.

    .. seealso:: :data:`numpy.arctan2`

    ''')


deg2rad = _core.create_ufunc(
    'cupy_deg2rad',
    ('e->e', 'f->f', 'd->d'),
    'out0 = in0 * (out0_type)(M_PI / 180)',
    doc='''Converts angles from degrees to radians elementwise.

    .. seealso:: :data:`numpy.deg2rad`, :data:`numpy.radians`

    ''')


rad2deg = _core.create_ufunc(
    'cupy_rad2deg',
    ('e->e', 'f->f', 'd->d'),
    'out0 = in0 * (out0_type)(180 / M_PI)',
    doc='''Converts angles from radians to degrees elementwise.

    .. seealso:: :data:`numpy.rad2deg`, :data:`numpy.degrees`

    ''')


@_core.fusion.fuse()
def _unwrap_correct(dd, discont):
    ddmod = cupy.mod(dd + numpy.pi, 2*numpy.pi) - numpy.pi
    cupy.copyto(ddmod, numpy.pi, where=(ddmod == -numpy.pi) & (dd > 0))
    ph_correct = ddmod - dd
    cupy.copyto(ph_correct, 0., where=cupy.abs(dd) < discont)
    return ph_correct


def unwrap(p, discont=None, axis=-1, *, period=2*numpy.pi):
    """Unwrap by taking the complement of large deltas w.r.t. the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than ``max(discont, period/2)``
    to their `period`-complementary values.

    For the default case where `period` is ``2 * pi`` and is `discont` is
    ``pi``, this unwraps a radian phase `p` such that adjacent differences
    are never greater than ``pi`` by adding ``2ki*pi`` for some
    integer ``k``.

    Args:
        p (cupy.ndarray): Input array.
        discont (float): Maximum discontinuity between values, default is
        ``period/2``. Values below ``period/2`` are treated as if they were
        ``period/2``. To have an effect different from the default, `discont`
        should be larger than ``period/2``.
        axis (int): Axis along which unwrap will operate, default is the last
            axis.
        period: float, optional
        Size of the range over which the input wraps. By default, it is
        ``2 pi``.
    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.unwrap`
    """

    p = cupy.asarray(p)
    nd = p.ndim
    dd = sumprod.diff(p, axis=axis)
    if discont is None:
        discont = period/2
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ph_correct = _unwrap_correct(dd, discont)
    dtype = numpy.result_type(dd.dtype, period)
    if numpy.core.numeric.issubdtype(dtype, numpy.core.numeric.integer):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = cupy.mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        # for `mask = (abs(dd) == period/2)`, the above line made
        # `ddmod[mask] == -period/2`. correct these such that
        # `ddmod[mask] == sign(dd[mask])*period/2`.
        ddmod[(ddmod == interval_low) & (dd > 0)] = interval_high

    up = cupy.array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + cupy.cumsum(ph_correct, axis=axis)
    return up


degrees = rad2deg
radians = deg2rad
