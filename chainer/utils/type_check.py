import numpy

from chainer import cuda


class TypeInfo(object):

    def __init__(self, name, index, shape, dtype):
        self.index = index
        #name = '{0} argument'.format(_make_ordinal(index + 1))
        name = '{0}[{1}]'.format(name, index)
        self.name = name
        self.shape = tuple(Shape(x, i, name) for i, x in enumerate(shape))
        self.dtype = DtypeExpr(dtype, name)
        self.ndim = Member(len(self.shape), name, 'ndim')

    def is_none(self):
        return self.dtype is None

    def __str__(self):
        return self.name


class TypeInfoTuple(tuple):

    def __init__(self, data):
        super(TypeInfoTuple, self).__init__(data)

    def _set_name(self, name):
        self.name = name

    def size(self):
        return Member(len(self), self.name, 'size')


def get_types(data, is_grad):
    assert(isinstance(data, tuple))

    if is_grad:
        name = 'out_types'
    else:
        name = 'in_types'

    info = TypeInfoTuple(
        get_type(name, i, x, is_grad) for i, x in enumerate(data))
    info._set_name(name)
    return info


def get_type(name, index, array, accept_none):
    if accept_none and array is None:
        # case that gradient is not given
        return TypeInfo(name, index, (), None)

    assert(isinstance(array, numpy.ndarray) or
           isinstance(array, cuda.GPUArray))
    return TypeInfo(name, index, array.shape, array.dtype)


def _wrap(x):
    if isinstance(x, IntExpr):
        return x
    elif isinstance(x, int):
        return IntConstant(x)
    else:
        assert(False)


def _make_un_operator(exp, priority, func):
    def f(x):
        return UnaryOperator(_wrap(x), exp, priority, func)
    return f


def _make_bin_operator(exp, priority, func):
    def f(x, y):
        return BinaryOperator(_wrap(x), _wrap(y),
                              exp, priority, func)
    return f


def _flip(f):
    return lambda x, y: f(y, x)


_add = _make_bin_operator('+', 4, int.__add__)
_sub = _make_bin_operator('-', 4, int.__sub__)
_mul = _make_bin_operator('*', 5, int.__mul__)
_floordiv = _make_bin_operator('//', 5, int.__floordiv__)
_mod = _make_bin_operator('%', 5, int.__mod__)
_pow = _make_bin_operator('**', 7, int.__mod__)
_lshift = _make_bin_operator('<<', 3, int.__lshift__)
_rshift = _make_bin_operator('>>', 3, int.__rshift__)
_and = _make_bin_operator('&', 2, int.__and__)
_xor = _make_bin_operator('^', 1, int.__xor__)
_or = _make_bin_operator('|', 0, int.__or__)


class IntExpr(object):

    def __init__(self, priority):
        self.priority = priority

    def should_be(self, expect):
        expect = _wrap(expect)

        actual_value = self.eval()
        expect_value = expect.eval()
        if actual_value != expect_value:
            raise InvalidType(
                '{0} == {1}'.format(self, expect),
                '{0} != {1}'.format(actual_value, expect_value))

    def eval(self):
        pass

    __add__ = _add
    __radd__ = _flip(_add)
    __sub__ = _sub
    __rsub__ = _flip(_sub)
    __mul__ = _mul
    __rmul__ = _flip(_mul)
    __floordiv__ = _floordiv
    __rfloordiv__ = _flip(_floordiv)
    __mod__ = _mod
    __rmod__ = _flip(_mod)
    __pow__ = _pow

    __lshift__ = _lshift
    __rshift__ = _rshift

    __and__ = _and
    __rand__ = _flip(_and)
    __xor__ = _xor
    __rxor__ = _flip(_xor)
    __or__ = _or
    __ror__ = _flip(_or)

    __neg__ = _make_un_operator('-', 6, int.__neg__)
    __pos__ = _make_un_operator('+', 6, int.__pos__)
    __invert__ = _make_un_operator('~', 6, int.__invert__)

    __lt__ = _make_bin_operator('<', -1, int.__lt__)
    __le__ = _make_bin_operator('<=', -1, int.__le__)
    __eq__ = _make_bin_operator('==', -1, int.__eq__)
    __ne__ = _make_bin_operator('!=', -1, int.__ne__)
    __gt__ = _make_bin_operator('>', -1, int.__gt__)
    __ge__ = _make_bin_operator('>', -1, int.__ge__)


class IntAtom(IntExpr):

    def __init__(self, value):
        super(IntAtom, self).__init__(8)
        self.value = value

    def eval(self):
        return self.value


class IntConstant(IntAtom):

    def __init__(self, value):
        super(IntConstant, self).__init__(value)

    def __str__(self):
        return str(self.value)


class IntVariable(IntAtom):

    def __init__(self, value, name):
        super(IntVariable, self).__init__(value)
        self.name = name

    def __str__(self):
        return self.name


class Shape(IntAtom):

    def __init__(self, value, index, name):
        super(Shape, self).__init__(value)
        self.name = name
        self.index = index

    def __str__(self):
        return '{0}.shape[{1}]'.format(self.name, self.index)


class Member(IntAtom):

    def __init__(self, value, obj, name):
        super(Member, self).__init__(value)
        self.obj = obj
        self.name = name

    def __str__(self):
        return '{0}.{1}'.format(self.obj, self.name)


class UnaryOperator(IntExpr):

    def __init__(self, term, exp, priority, func):
        super(UnaryOperator, self).__init__(priority)
        self.term = term
        self.exp = exp
        self.func = func

    def eval(self):
        return self.func(self.term.eval())

    def __str__(self):
        exp = str(self.term)
        if self.term.priority < self.priority:
            exp = '(' + exp + ')'

        return self.exp + exp


class BinaryOperator(IntExpr):

    def __init__(self, lhs, rhs, exp, priority, func):
        super(BinaryOperator, self).__init__(priority)
        self.lhs = lhs
        self.rhs = rhs
        self.exp = exp
        self.func = func

    def eval(self):
        return self.func(self.lhs.eval(), self.rhs.eval())

    def __str__(self):
        left = str(self.lhs)
        if self.priority > self.lhs.priority:
            left = '(' + left + ')'

        right = str(self.rhs)
        # As infix operators are left-associative, we need to append parens
        # when rhs has the same priority
        #  e.g. x << (y << z) != x << y << z
        if self.priority >= self.rhs.priority:
            right = '(' + right + ')'

        return '{0} {2} {1}'.format(left, right, self.exp)


class DtypeExpr(object):

    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

    def should_be(self, expect):
        if isinstance(expect, DtypeExpr):
            expect_value = expect.eval()
        else:
            expect_value = expect

        if not (self.eval() == expect):
            raise InvalidType(
                '{0} == {1}'.format(self, expect),
                '{0} != {1}'.format(self.dtype, expect_value))

    def kind_should_be(self, expect):
        if not (self.eval().kind == expect):
            raise InvalidType(
                '{0}.kind == {1}'.format(self, expect),
                '{0} != {1}'.format(self.dtype.kind, expect))

    def eval(self):
        return self.dtype

    def __str__(self):
        return '{0}.dtype'.format(self.name)


class InvalidType(Exception):

    def __init__(self, expect, actual):
        msg = 'Expect: {0}\nActual: {1}'.format(expect, actual)
        super(InvalidType, self).__init__(msg)

        self.expect = expect
        self.actual = actual
