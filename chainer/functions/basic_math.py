from chainer import cuda, Function, Variable

class Neg(Function):
    def forward(self, x):
        return -x[0],

    def backward(self, x, gy):
        return -gy[0],

def neg(x):  # -x
    return Neg()(x)


class Add(Function):
    def forward(self, x):
        return x[0] + x[1],

    def backward(self, x, gy):
        return gy[0], gy[0]

class AddConstant(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return x[0] + self.value,

    def backward(self, x, gy):
        return gy[0],

def add(lhs, rhs):  # lhs + rhs
    if type(rhs) != Variable:
        return AddConstant(rhs)(lhs)
    return Add()(lhs, rhs)


class Sub(Function):
    def forward(self, x):
        return x[0] - x[1],

    def backward(self, x, gy):
        return gy[0], -gy[0]

def sub(lhs, rhs):  # lhs - rhs
    if type(rhs) != Variable:
        return AddConstant(-rhs)(lhs)
    return Sub()(lhs, rhs)


class SubFromConstant(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return self.value - x[0],

    def backward(self, x, gy):
        return -gy[0],

def rsub(lhs, rhs):  # rhs - lhs
    if type(rhs) != Variable:
        return SubFromConstant(rhs)(lhs)
    return Sub()(rhs, lhs)


class Mul(Function):
    def forward(self, x):
        return x[0] * x[1],

    def backward_cpu(self, x, gy):
        return gy[0] * x[1], gy[0] * x[0]

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            'float* gx0, float* gx1, const float* x0, const float* x1, const float* gy',
            '''gx0[i] = gy[i] * x1[i];
               gx1[i] = gy[i] * x0[i];''',
            'mul_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1

class MulConstant(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return self.value * x[0],

    def backward(self, x, gy):
        return self.value * gy[0],

def mul(lhs, rhs):  # lhs * rhs
    if type(rhs) != Variable:
        return MulConstant(rhs)(lhs)
    return Mul()(lhs, rhs)


class Div(Function):
    def forward(self, x):
        return x[0] / x[1],

    def backward_cpu(self, x, gy):
        gx0 = gy[0] / x[1]
        return gx0, -gx0 * x[0] / x[1]

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            'float* gx0, float* gx1, const float* x0, const float* x1, const float* gy',
            '''gx0[i] = gy[i] / x1[i];
               gx1[i] = -gx0[i] * x0[i] / x1[i];''',
            'div_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1

def div(lhs, rhs):  # lhs / rhs
    if type(rhs) != Variable:
        return MulConstant(1. / rhs)(lhs)
    return Div()(lhs, rhs)

class DivFromConstant(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return self.value / x[0],

    def backward_cpu(self, x, gy):
        return -self.value * gy[0] / (x[0] ** 2),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        cuda.elementwise(
            'float* gx, const float* x, const float* gy, float value',
            'gx[i] = -value * gy[i] / (x[i] * x[i])',
            'div_from_const_bwd')(gx, x[0], gy[0], self.value)
        return gx,

def rdiv(lhs, rhs):  # rhs / lhs
    if type(rhs) != Variable:
        return DivFromConstant(rhs)(lhs)
    return Div()(rhs, lhs)


class PowVarConst(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return x[0] ** self.value,

    def backward_cpu(self, x, gy):
        return self.value * (x[0] ** (self.value - 1)) * gy[0],

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        cuda.elementwise(
            'float* gx, const float* x, const float* gy, float value',
            'gx[i] = value * __powf(x[i], value - 1) * gy[i]',
            'pow_var_const_bwd')(gx, x[0], gy[0], self.value)
        return gx,

def pow(lhs, rhs):  # lhs ** rhs
    # TODO(beam2d): Support const ** var and var ** var
    if type(rhs) == Variable:
        raise NotImplementedError()
    return PowVarConst(rhs)(lhs)


# Variable operators
Variable.__neg__  = neg
Variable.__add__  = add
Variable.__radd__ = add
Variable.__sub__  = sub
Variable.__rsub__ = rsub
Variable.__mul__  = mul
Variable.__rmul__ = mul
Variable.__div__  = div
Variable.__rdiv__ = rdiv
Variable.__pow__  = pow
