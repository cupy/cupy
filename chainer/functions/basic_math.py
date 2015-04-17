from chainer import Function, Variable

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

    def backward(self, x, gy):
        # TODO(beam2d): Unify to one kernel
        return gy[0] * x[1], gy[0] * x[0]

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

    def backward(self, x, gy):
        # TODO(beam2d): Unify to one kernel
        gx0 = gy[0] / x[1]
        return gx0, -gx0 * x[0] / x[1]

def div(lhs, rhs):  # lhs / rhs
    if type(rhs) != Variable:
        return MulConstant(1. / rhs)(lhs)
    return Div()(lhs, rhs)

class DivFromConstant(Function):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return self.value / x[0],

    def backward(self, x, gy):
        # TODO(beam2d): Unify to one kernel
        return -self.value * gy[0] / (x[0] ** 2),

def rdiv(lhs, rhs):  # rhs / lhs
    if type(rhs) != Variable:
        return DivFromConstant(rhs)(lhs)
    return Div()(rhs, lhs)


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
