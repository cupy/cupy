import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class Sin(function.Function):

    @property
    def label(self):
        return 'sin'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sin(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.cos(x[0]))
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = cos(x) * gy', 'sin_bwd'
        )(x[0], gy[0])
        return gx,


def sin(x):
    """Elementwise sin function."""
    return Sin()(x)


class Cos(function.Function):

    @property
    def label(self):
        return 'cos'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.cos(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.sin(x[0]))
        numpy.negative(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = -sin(x) * gy', 'cos_bwd'
        )(x[0], gy[0])
        return gx,


def cos(x):
    """Elementwise cos function."""
    return Cos()(x)


class Tan(function.Function):

    @property
    def label(self):
        return 'tan'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.tan(x[0])),

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = utils.force_array(xp.cos(x[0]))
        xp.square(gx, out=gx)
        xp.reciprocal(gx, out=gx)
        gx *= gy[0]
        return gx,


def tan(x):
    """Elementwise tan function."""
    return Tan()(x)


class Arcsin(function.Function):

    @property
    def label(self):
        return 'arcsin'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.arcsin(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.square(x[0]))
        numpy.negative(gx, out=gx)
        gx += 1
        numpy.sqrt(gx, out=gx)
        numpy.reciprocal(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = rsqrt((T)1.0 - x * x) * gy',
            'arcsin_bwd'
        )(x[0], gy[0])
        return gx,


def arcsin(x):
    """Elementwise arcsine function.

    .. math::
       y_i = \\arcsin x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arcsin()(x)


class Arccos(function.Function):

    @property
    def label(self):
        return 'arccos'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.arccos(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.square(x[0]))
        numpy.negative(gx, out=gx)
        gx += 1
        numpy.sqrt(gx, out=gx)
        numpy.reciprocal(gx, out=gx)
        numpy.negative(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = -rsqrt((T)1.0 - x * x) * gy',
            'arccos_bwd'
        )(x[0], gy[0])
        return gx,


def arccos(x):
    """Elementwise arccosine function.

    .. math::
       y_i = \\arccos x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arccos()(x)


class Arctan(function.Function):

    @property
    def label(self):
        return 'arctan'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.arctan(x[0])),

    def backward_cpu(self, x, gy):
        gx = utils.force_array(numpy.square(x[0]))
        gx += 1
        numpy.reciprocal(gx, out=gx)
        gx *= gy[0]
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = (T)1.0 / ((T)1.0 + x * x) * gy',
            'arctan_bwd'
        )(x[0], gy[0])
        return gx,


def arctan(x):
    """Elementwise arctangent function.

    .. math::
       y_i = \\arctan x_i.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arctan()(x)
