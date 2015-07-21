import math
from numbers import Number

import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


# ------------------------------------------------------------------------------
# Arithmetic
# ------------------------------------------------------------------------------

def _convert_value_to_string(value):
    if isinstance(value, variable.Variable):
        value = value.data

    if isinstance(value, float):
        return str(value)
    elif isinstance(value, (numpy.ndarray, cuda.GPUArray)):
        return 'constant array'
    else:
        raise ValueError(
            'value must be float, ndarray, GPUArray, or Variable')


def _force_type(dtype, value):
    if numpy.isscalar(value):
        return dtype.type(value)
    else:
        return value


class Neg(function.Function):

    @property
    def label(self):
        return '__neg__'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(-x[0]),

    def backward(self, x, gy):
        return utils.force_array(-gy[0]),


def neg(x):  # -x
    return Neg()(x)


class Absolute(function.Function):

    @property
    def label(self):
        return '|_|'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(abs(x[0])),

    def backward_cpu(self, x, gy):
        return utils.force_array(numpy.sign(x[0]) * gy[0]),

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        cuda.elementwise(
            'float* gx0, const float* x0, const float* gy',
            'gx0[i] = ((x0[i] > 0) - (x0[i] < 0)) * gy[i]',
            'abs_bwd')(gx0, x[0], gy[0])
        return gx0,


def absolute(x):
    return Absolute()(x)


class Add(function.Function):

    @property
    def label(self):
        return '_ + _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        y = utils.force_array(x[0] + x[1])
        return y,

    def backward(self, x, gy):
        return gy[0], gy[0]


class AddConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ + %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(x[0] + _force_type(x[0].dtype, self.value)),

    def backward(self, x, gy):
        return gy[0],


def add(lhs, rhs):  # lhs + rhs
    if isinstance(rhs, variable.Variable):
        return Add()(lhs, rhs)
    return AddConstant(rhs)(lhs)


class Sub(function.Function):

    @property
    def label(self):
        return '_ - _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] - x[1]),

    def backward(self, x, gy):
        return gy[0], utils.force_array(-gy[0])


def sub(lhs, rhs):  # lhs - rhs
    if isinstance(rhs, variable.Variable):
        return Sub()(lhs, rhs)
    return AddConstant(-rhs)(lhs)


class SubFromConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s - _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(_force_type(x[0].dtype, self.value) - x[0]),

    def backward(self, x, gy):
        return utils.force_array(-gy[0]),


def rsub(lhs, rhs):  # rhs - lhs
    if isinstance(rhs, variable.Variable):
        return Sub()(rhs, lhs)
    return SubFromConstant(rhs)(lhs)


class Mul(function.Function):

    @property
    def label(self):
        return '_ * _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] * x[1]),

    def backward_cpu(self, x, gy):
        return utils.force_array(gy[0] * x[1]), utils.force_array(gy[0] * x[0])

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            '''
               float* gx0, float* gx1, const float* x0, const float* x1,
               const float* gy
            ''', '''
               gx0[i] = gy[i] * x1[i];
               gx1[i] = gy[i] * x0[i];
            ''', 'mul_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1


class MulConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ * %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(_force_type(x[0].dtype, self.value) * x[0]),

    def backward(self, x, gy):
        return utils.force_array(_force_type(gy[0].dtype, self.value) * gy[0]),


def mul(lhs, rhs):  # lhs * rhs
    if isinstance(rhs, variable.Variable):
        return Mul()(lhs, rhs)
    return MulConstant(rhs)(lhs)


class Div(function.Function):

    @property
    def label(self):
        return '_ / _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] / x[1]),

    def backward_cpu(self, x, gy):
        gx0 = utils.force_array(gy[0] / x[1])
        return gx0, utils.force_array(-gx0 * x[0] / x[1])

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            '''
               float* gx0, float* gx1, const float* x0, const float* x1,
               const float* gy
            ''', '''
               gx0[i] = gy[i] / x1[i];
               gx1[i] = -gx0[i] * x0[i] / x1[i];
            ''', 'div_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1


def div(lhs, rhs):  # lhs / rhs
    if isinstance(rhs, variable.Variable):
        return Div()(lhs, rhs)
    return MulConstant(1. / rhs)(lhs)


class DivFromConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ / %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(_force_type(x[0].dtype, self.value) / x[0]),

    def backward_cpu(self, x, gy):
        value = _force_type(gy[0].dtype, self.value)
        return utils.force_array(-value * gy[0] / (x[0] ** 2)),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        if isinstance(self.value, Number):
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float value
                ''',
                'gx[i] = -value * gy[i] / (x[i] * x[i])',
                'div_from_const_bwd')(gx, x[0], gy[0], self.value)
        else:
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float* value
                ''',
                'gx[i] = -value[i] * gy[i] / (x[i] * x[i])',
                'div_from_const_bwd')(gx, x[0], gy[0], self.value)
        return gx,


def rdiv(lhs, rhs):  # rhs / lhs
    if isinstance(rhs, variable.Variable):
        return Div()(rhs, lhs)
    return DivFromConstant(rhs)(lhs)


class PowVarVar(function.Function):

    @property
    def label(self):
        return '_ ** _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, x):
        self.y = utils.force_array(x[0] ** x[1])
        return self.y,

    def forward_gpu(self, x):
        return x[0] ** x[1],

    def backward_cpu(self, x, gy):
        one = x[1].dtype.type(1)
        gx0 = utils.force_array(x[1] * (x[0] ** (x[1] - one)) * gy[0])
        gx1 = utils.force_array(numpy.log(x[0]) * self.y * gy[0])
        return gx0, gx1

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            '''
               float* gx0, float* gx1, const float* x0, const float* x1,
               const float* gy
            ''', '''
               gx0[i] = x1[i] * powf(x0[i], x1[i] - 1) * gy[i];
               gx1[i] = __logf(x0[i]) * powf(x0[i], x1[i]) * gy[i];
            ''', 'pow_var_var_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1


class PowVarConst(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ ** %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        return utils.force_array(x[0] ** _force_type(x[0].dtype, self.value)),

    def backward_cpu(self, x, gy):
        val_1 = _force_type(x[0].dtype, self.value - 1)
        gx = _force_type(x[0].dtype, self.value) * (x[0] ** val_1) * gy[0]
        return utils.force_array(gx),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        if isinstance(self.value, Number):
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float value
                ''',
                'gx[i] = value * powf(x[i], value - 1) * gy[i]',
                'pow_var_const_bwd')(gx, x[0], gy[0], self.value)
        else:
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float* value
                ''',
                'gx[i] = value[i] * powf(x[i], value[i] - 1) * gy[i]',
                'pow_var_const_bwd')(gx, x[0], gy[0], self.value)
        return gx,


def pow(lhs, rhs):  # lhs ** rhs
    if isinstance(rhs, variable.Variable):
        return PowVarVar()(lhs, rhs)
    return PowVarConst(rhs)(lhs)


class PowConstVar(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s ** _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward_cpu(self, x):
        self.y = utils.force_array(_force_type(x[0].dtype, self.value) ** x[0])
        return self.y,

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        if isinstance(self.value, Number):
            cuda.elementwise('float* y, const float* x, const float value',
                             'y[i] = powf(value, x[i])',
                             'pow_const_var_fwd')(y, x[0], self.value)
        else:
            cuda.elementwise('float* y, const float* x, const float *value',
                             'y[i] = powf(value[i], x[i])',
                             'pow_const_var_fwd')(y, x[0], self.value)
        return y,

    def backward_cpu(self, x, gy):
        value = _force_type(gy[0].dtype, self.value)
        return utils.force_array(numpy.log(value) * self.y * gy[0]),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        if isinstance(self.value, Number):
            logv = math.log(self.value)
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float value, const float logv
                ''',
                'gx[i] = logv * powf(value, x[i]) * gy[i]',
                'pow_const_var_bwd')(gx, x[0], gy[0], self.value, logv)
        else:
            cuda.elementwise(
                '''
                   float* gx, const float* x, const float* gy,
                   const float* value
                ''',
                'gx[i] = __logf(value[i]) * powf(value[i], x[i]) * gy[i]',
                'pow_const_var_bwd')(gx, x[0], gy[0], self.value)
        return gx,


def rpow(lhs, rhs):  # rhs ** lhs
    if isinstance(rhs, variable.Variable):
        return PowVarVar()(rhs, lhs)
    return PowConstVar(rhs)(lhs)


def install_variable_arithmetics():
    variable.Variable.__neg__ = neg
    variable.Variable.__abs__ = absolute
    variable.Variable.__add__ = add
    variable.Variable.__radd__ = add
    variable.Variable.__sub__ = sub
    variable.Variable.__rsub__ = rsub
    variable.Variable.__mul__ = mul
    variable.Variable.__rmul__ = mul
    variable.Variable.__div__ = div
    variable.Variable.__truediv__ = div
    variable.Variable.__rdiv__ = rdiv
    variable.Variable.__rtruediv__ = rdiv
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow

# ------------------------------------------------------------------------------
# Special functions
# ------------------------------------------------------------------------------


class Exp(function.Function):

    @property
    def label(self):
        return 'exp'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward_cpu(self, x):
        self.y = utils.force_array(numpy.exp(x[0]))
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.cumath.exp(x[0])
        return self.y,

    def backward(self, x, gy):
        return utils.force_array(self.y * gy[0]),


def exp(x):
    """Elementwise exponential function."""
    return Exp()(x)


class Log(function.Function):

    @property
    def label(self):
        return 'log'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward_cpu(self, x):
        return utils.force_array(numpy.log(x[0])),

    def forward_gpu(self, x):
        return cuda.cumath.log(x[0]),

    def backward(self, x, gy):
        return utils.force_array(gy[0] / x[0]),


def log(x):
    """Elementwise natural logarithm function."""
    return Log()(x)
