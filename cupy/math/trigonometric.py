from cupy import elementwise
from cupy.math import ufunc

sin = ufunc.create_math_ufunc('sin', 1)
cos = ufunc.create_math_ufunc('cos', 1)
tan = ufunc.create_math_ufunc('tan', 1)
arcsin = ufunc.create_math_ufunc('asin', 1, 'arcsin')
arccos = ufunc.create_math_ufunc('acos', 1, 'arccos')
arctan = ufunc.create_math_ufunc('atan', 1, 'arctan')
hypot = ufunc.create_math_ufunc('hypot', 2)
arctan2 = ufunc.create_math_ufunc('atan2', 2, 'arctan2')


deg2rad = elementwise.create_ufunc(
    'cupy_deg2rad',
    ['e->e', 'f->f', 'd->d'],
    'out0 = in0 * (out0_type)(M_PI / 180)')


rad2deg = elementwise.create_ufunc(
    'cupy_rad2deg',
    ['e->e', 'f->f', 'd->d'],
    'out0 = in0 * (out0_type)(180 / M_PI)')


def unwrap(p, discont=3.141592653589793, axis=-1, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


degrees = rad2deg
radians = deg2rad
