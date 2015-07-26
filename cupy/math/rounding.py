from cupy.math import ufunc


def around(a, decimals=0, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


# TODO(beam2d): Implement it
# round_ = around


rint = ufunc.create_math_ufunc('rint', 1)


def fix(x, y=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


floor = ufunc.create_math_ufunc('floor', 1)
ceil = ufunc.create_math_ufunc('ceil', 1)
trunc = ufunc.create_math_ufunc('trunc', 1)
