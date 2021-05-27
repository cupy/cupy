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


def unwrap(p, discont=numpy.pi, axis=-1):
    """Unwrap by changing deltas between values to 2*pi complement.

    Args:
        p (cupy.ndarray): Input array.
        discont (float): Maximum discontinuity between values, default is
            ``pi``.
        axis (int): Axis along which unwrap will operate, default is the last
            axis.
    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.unwrap`
    """

    p = cupy.asarray(p)
    nd = p.ndim
    dd = sumprod.diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ph_correct = _unwrap_correct(dd, discont)
    up = cupy.array(p, copy=True, dtype='d')
    up[slice1] = p[slice1] + cupy.cumsum(ph_correct, axis=axis)
    return up


degrees = rad2deg
radians = deg2rad
