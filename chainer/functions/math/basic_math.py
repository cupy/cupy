import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


def _convert_value_to_string(value):
    if isinstance(value, variable.Variable):
        value = value.data

    if numpy.isscalar(value):
        if value < 0:
            return '({})'.format(value)
        else:
            return str(value)
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return 'constant array'
    else:
        raise ValueError(
            'value must be scalar, ndarray, or Variable')


def _check_constant_type(value):
    if numpy.isscalar(value):
        return
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return
    else:
        raise ValueError(
            'value must be scalar, ndarray, or Variable')


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
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward(self, x):
        return utils.force_array(abs(x[0])),

    def backward_cpu(self, x, gy):
        return utils.force_array(numpy.sign(x[0]) * gy[0]),

    def backward_gpu(self, x, gy):
        gx0 = cuda.elementwise(
            'T x0, T gy', 'T gx0',
            'gx0 = ((x0 > 0) - (x0 < 0)) * gy',
            'abs_bwd')(x[0], gy[0])
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
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(x[0] + value),

    def backward(self, x, gy):
        return gy[0],


def add(lhs, rhs):  # lhs + rhs
    if isinstance(rhs, variable.Variable):
        return Add()(lhs, rhs)
    _check_constant_type(rhs)
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
    _check_constant_type(rhs)
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
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(value - x[0]),

    def backward(self, x, gy):
        return utils.force_array(-gy[0]),


def rsub(lhs, rhs):  # rhs - lhs
    if isinstance(rhs, variable.Variable):
        return Sub()(rhs, lhs)
    _check_constant_type(rhs)
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

    def backward(self, x, gy):
        return utils.force_array(gy[0] * x[1]), utils.force_array(gy[0] * x[0])


class MulConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ * %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(value * x[0]),

    def backward(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(value * gy[0]),


def mul(lhs, rhs):  # lhs * rhs
    if isinstance(rhs, variable.Variable):
        return Mul()(lhs, rhs)
    _check_constant_type(rhs)
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
        return cuda.elementwise(
            'T x0, T x1, T gy',
            'T gx0, T gx1',
            '''
               gx0 = gy / x1;
               gx1 = -gx0 * x0 / x1;
            ''', 'div_bwd')(x[0], x[1], gy[0])


def div(lhs, rhs):  # lhs / rhs
    if isinstance(rhs, variable.Variable):
        return Div()(lhs, rhs)
    _check_constant_type(rhs)
    return MulConstant(1. / rhs)(lhs)


class DivFromConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ / %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward(self, x):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(value / x[0]),

    def backward_cpu(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(-value * gy[0] / (x[0] ** 2)),

    def backward_gpu(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        gx = cuda.elementwise('T x, T gy, T value', 'T gx',
                              'gx = -value * gy / (x * x)',
                              'div_from_const_bwd')(x[0], gy[0], value)
        return gx,


def rdiv(lhs, rhs):  # rhs / lhs
    if isinstance(rhs, variable.Variable):
        return Div()(rhs, lhs)
    _check_constant_type(rhs)
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
        return cuda.elementwise(
            'T x0, T x1, T gy', 'T gx0, T gx1',
            '''
               gx0 = x1 * pow(x0, x1 - 1) * gy;
               gx1 = log(x0) * pow(x0, x1) * gy;
            ''', 'pow_var_var_bwd')(x[0], x[1], gy[0])


class PowVarConst(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ ** %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward(self, x):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(x[0] ** value),

    def backward_cpu(self, x, gy):
        val_1 = utils.force_type(x[0].dtype, self.value - 1)
        gx = utils.force_type(x[0].dtype, self.value) * (x[0] ** val_1) * gy[0]
        return utils.force_array(gx),

    def backward_gpu(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        gx = cuda.elementwise(
            'T x, T gy, T value', 'T gx',
            'gx = value * pow(x, value - 1) * gy',
            'pow_var_const_bwd')(x[0], gy[0], value)
        return gx,


def pow(lhs, rhs):  # lhs ** rhs
    if isinstance(rhs, variable.Variable):
        return PowVarVar()(lhs, rhs)
    _check_constant_type(rhs)
    return PowVarConst(rhs)(lhs)


class PowConstVar(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s ** _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward(self, x):
        value = utils.force_type(x[0].dtype, self.value)
        self.y = utils.force_array(value ** x[0])
        return self.y,

    def backward_cpu(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        return utils.force_array(numpy.log(value) * self.y * gy[0]),

    def backward_gpu(self, x, gy):
        value = utils.force_type(x[0].dtype, self.value)
        gx = cuda.elementwise(
            'T x, T gy, T value', 'T gx',
            'gx = log(value) * pow(value, x) * gy',
            'pow_const_var_bwd')(x[0], gy[0], value)
        return gx,


def rpow(lhs, rhs):  # rhs ** lhs
    if isinstance(rhs, variable.Variable):
        return PowVarVar()(rhs, lhs)
    _check_constant_type(rhs)
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
