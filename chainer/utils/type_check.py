import operator

import numpy

from chainer import cuda


class TypeInfo(object):

    def __init__(self, name, index, shape, dtype):
        self.index = index
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

    def size(self):
        return Member(len(self), self.name, 'size')


def get_types(data, is_grad):
    assert(isinstance(data, tuple))

    if is_grad:
        name = 'out_types'
    else:
        name = 'in_types'

    info = TypeInfoTuple(
        _get_type(name, i, x, is_grad) for i, x in enumerate(data))
    # I don't know a method to set an atribute in an initializer of tuple.
    info.name = name
    return info


def _get_type(name, index, array, accept_none):
    if accept_none and array is None:
        # case that gradient is not given
        return TypeInfo(name, index, (), None)

    assert(isinstance(array, numpy.ndarray) or
           isinstance(array, cuda.GPUArray))
    return TypeInfo(name, index, array.shape, array.dtype)


def _make_un_operator(exp, priority, func):
    def f(x):
        return IntUnaryOperator(priority, x, exp, func)
    return f


def _make_bin_operator(exp, priority, func):
    def f(x, y):
        return IntBinaryOperator(priority, x, y, exp, func)
    return f


def _make_bool_operator(exp, inv, func):
    def f(x, y):
        return BoolBinaryOperator(x, y, exp, inv, func)
    return f


def _flip(f):
    return lambda x, y: f(y, x)


class Expr(object):

    def __init__(self, priority):
        self.priority = priority

    def eval(self):
        raise NotImplemented()

    def __nonzero__(self):
        msg = ('Don\'t convert Expr to bool. '
               'Please call Expr.eval method to evaluate expression.')
        raise RuntimeError(msg)

    def __bool__(self):
        self.__nonzero__()

    __eq__ = _make_bool_operator('==', '!=', operator.__eq__)
    __ne__ = _make_bool_operator('!=', '==', operator.__ne__)
    __lt__ = _make_bool_operator('<', '>=', operator.__lt__)
    __le__ = _make_bool_operator('<=', '>', operator.__le__)
    __gt__ = _make_bool_operator('>', '<=', operator.__gt__)
    __ge__ = _make_bool_operator('>=', '<', operator.__ge__)


class Int(object):

    __add__ = _make_bin_operator('+', 4, int.__add__)
    __radd__ = _flip(__add__)
    __sub__ = _make_bin_operator('-', 4, int.__sub__)
    __rsub__ = _flip(__sub__)
    __mul__ = _make_bin_operator('*', 5, int.__mul__)
    __rmul__ = _flip(__mul__)
    __floordiv__ = _make_bin_operator('//', 5, int.__floordiv__)
    __rfloordiv__ = _flip(__floordiv__)
    __mod__ = _make_bin_operator('%', 5, int.__mod__)
    __rmod__ = _flip(__mod__)
    __pow__ = _make_bin_operator('**', 7, int.__mod__)

    __lshift__ = _make_bin_operator('<<', 3, int.__lshift__)
    __rlshift__ = _flip(__lshift__)
    __rshift__ = _make_bin_operator('>>', 3, int.__rshift__)
    __rrshift__ = _flip(__rshift__)

    __and__ = _make_bin_operator('&', 2, int.__and__)
    __rand__ = _flip(__and__)
    __xor__ = _make_bin_operator('^', 1, int.__xor__)
    __rxor__ = _flip(__xor__)
    __or__ = _make_bin_operator('|', 0, int.__or__)
    __ror__ = _flip(__or__)

    __neg__ = _make_un_operator('-', 6, int.__neg__)
    __pos__ = _make_un_operator('+', 6, int.__pos__)
    __invert__ = _make_un_operator('~', 6, int.__invert__)


class Atom(Expr):

    def __init__(self, value):
        super(Atom, self).__init__(8)
        self.value = value

    def eval(self):
        return self.value


class IntAtom(Atom, Int):

    def __init__(self, value):
        Atom.__init__(self, value)


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


class UnaryOperator(Expr):

    def __init__(self, priority, term, exp, func):
        super(UnaryOperator, self).__init__(priority)
        self.term = term
        self.exp = exp
        self.func = func

    def eval(self):
        return self.func(self.term.eval())

    def __str__(self):
        exp = str(self.term)
        if isinstance(self.term, Expr) and self.term.priority < self.priority:
            exp = '(' + exp + ')'

        return self.exp + exp


class IntUnaryOperator(UnaryOperator, Int):

    def __init__(self, priority, term, exp, func):
        UnaryOperator.__init__(self, priority, term, exp, func)


class BinaryOperator(Expr):

    def __init__(self, priority, lhs, rhs, exp, func):
        super(BinaryOperator, self).__init__(priority)
        self.lhs = lhs
        self.rhs = rhs
        self.exp = exp
        self.func = func

    def eval(self):
        left = self.eval_left()
        right = self.eval_right()
        return self.func(left, right)

    def eval_left(self):
        if isinstance(self.lhs, Expr):
            return self.lhs.eval()
        else:
            return self.lhs

    def eval_right(self):
        if isinstance(self.rhs, Expr):
            return self.rhs.eval()
        else:
            return self.rhs

    def __str__(self):
        left = str(self.lhs)
        if isinstance(self.lhs, Expr) and self.priority > self.lhs.priority:
            left = '(' + left + ')'

        right = str(self.rhs)
        # As infix operators are left-associative, we need to append parens
        # when rhs has the same priority
        #  e.g. x << (y << z) != x << y << z
        if isinstance(self.rhs, Expr) and self.priority >= self.rhs.priority:
            right = '(' + right + ')'

        return '{0} {2} {1}'.format(left, right, self.exp)


class IntBinaryOperator(BinaryOperator, Int):

    def __init__(self, priority, lhs, rhs, exp, func):
        BinaryOperator.__init__(self, priority, lhs, rhs, exp, func)


class Bool(object):

    def expect(self):
        raise NotImplemented()


class BoolBinaryOperator(BinaryOperator, Bool):

    def __init__(self, lhs, rhs, exp, inv, func):
        BinaryOperator.__init__(self, -1, lhs, rhs, exp, func)
        self.inv = inv

    def expect(self):
        left = self.eval_left()
        right = self.eval_right()

        if not self.func(left, right):
            raise InvalidType(
                '{0} {1} {2}'.format(self.lhs, self.exp, self.rhs),
                '{0} {1} {2}'.format(left, self.inv, right))


class DtypeExpr(Atom):

    def __init__(self, dtype, name):
        Atom.__init__(self, dtype)
        self.name = name

    def __str__(self):
        return '{0}.dtype'.format(self.name)


class InvalidType(Exception):

    def __init__(self, expect, actual):
        msg = 'Expect: {0}\nActual: {1}'.format(expect, actual)
        super(InvalidType, self).__init__(msg)

        self.expect = expect
        self.actual = actual


def expect(*bool_exprs):
    for expr in bool_exprs:
        assert isinstance(expr, Bool)
        expr.expect()
