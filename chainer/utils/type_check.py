import operator

import numpy

from chainer import cuda


class TypeInfo(object):

    """Type information of an input/gradient array.

    It contains type information of an array, such as shape of array and
    number of dimension. All information is representend as :class:`Expr`.
    So you can easily check its validity.
    This information is independent on CPU or GPU array.
    """

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

    """Type information of input/gradient tuples.

    It is a sub-class of tuple containing :class:`TypeInfo`. i-th element of
    this object contains type information of i-th input data.
    """

    def size(self):
        """Returns an expression representing its length.

        Returns:
            :class:`Expr` object representig length of the tulple.
        """
        return Member(len(self), self.name, 'size')


def get_types(data, name, accept_none):
    assert(isinstance(data, tuple))

    info = TypeInfoTuple(
        _get_type(name, i, x, accept_none) for i, x in enumerate(data))
    # I don't know a method to set an attribute in an initializer of tuple.
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


def _make_bin_operator(exp, priority, func, right_associative=False):
    def f(x, y):
        return IntBinaryOperator(priority, x, y, exp, func, right_associative)
    return f


def _make_bool_operator(exp, inv, func):
    def f(x, y):
        return BoolBinaryOperator(x, y, exp, inv, func)
    return f


def _flip(f):
    return lambda x, y: f(y, x)


class Expr(object):

    """Abstract syntax tree of an expression.

    It represents abstrat syntax tree, and isn't a value. You can get its
    actual value with :meth:`eval` function, and can get syntax representation
    with :meth:`__str__` method.
    Comparison operators (e.g. `==`) generates a new :class:`Expr` object
    which represent a result of comparison between two expression.

    .. admonition:: Example

       Let ``x`` and ``y`` are instances of :class:`Expr`, then

       >>> c = (x == y)

       is also an instance of :class:`Expr`. To evaluate and get its value,
       call :meth:`eval` method::

       >>> c.eval()
       True   # when x.eval() == y.eval()

       Call ``str` method to get representation of the original equaltion::

       >>> str(c)
       'x + y'   # when str(x) == 'x' and str(y) == 'y'

       You can actually compare an expression with a value::

       >>> (x == 1).eval()

       Note that you can't use boolean operators such as ``and``, as they try
       to cast expressions to a boolean values::

       >>> x == y and y == z  # Raises an error

    """

    def __init__(self, priority):
        self.priority = priority

    def eval(self):
        """Evaluate the tree to get actual value.

        Behavior of this functions is depends on an implementation class.
        For example, a binary operator `+` calls `__add__` function with two
        results of :meth:`eval` funciton.
        """
        raise NotImplementedError()

    def __nonzero__(self):
        # When a user calls a boolean operator like `(x == y and z == w)`,
        # `and` operator evaluate the both expressions and returns the last
        # result `z == w`.
        # So, `(x == y and z == w).expect()` only checks `z == w`. It is
        # confusing.
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

    # Please refer the Python document to know priority of operators.
    # https://docs.python.org/3.4/reference/expressions.html

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
    # Only '**' operator is right-associative
    __pow__ = _make_bin_operator('**', 7, int.__mod__, right_associative=True)

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

    def __init__(self, priority, lhs, rhs, exp, func, right_associative=False):
        super(BinaryOperator, self).__init__(priority)
        self.lhs = lhs
        self.rhs = rhs
        self.exp = exp
        self.func = func
        self.right_associative = right_associative

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
        # When an infix operator is left-associative, we need to append parens
        # when rhs has the same priority
        #  e.g. x << (y << z) != x << y << z

        left = str(self.lhs)
        if isinstance(self.lhs, Expr) and (
                self.priority > self.lhs.priority or
                (self.right_associative and
                 self.priority == self.lhs.priority)):
            left = '(' + left + ')'

        right = str(self.rhs)
        if isinstance(self.rhs, Expr) and (
                self.priority > self.rhs.priority or
                (not self.right_associative and
                 self.priority == self.rhs.priority)):
            right = '(' + right + ')'

        return '{0} {2} {1}'.format(left, right, self.exp)


class IntBinaryOperator(BinaryOperator, Int):

    def __init__(self, priority, lhs, rhs, exp, func, right_associative=False):
        BinaryOperator.__init__(self, priority, lhs, rhs, exp, func,
                                right_associative)


class Bool(object):

    def expect(self):
        raise NotImplementedError()


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
    """Raised when types of data for forward/backwar are invalid.

    """
    def __init__(self, expect, actual):
        msg = 'Expect: {0}\nActual: {1}'.format(expect, actual)
        super(InvalidType, self).__init__(msg)

        self.expect = expect
        self.actual = actual


def expect(*bool_exprs):
    """Evaluate and test all given expressions.

    This function evaluate given boolean expressions in order. When an
    expression is evaluated as `False`, that means the given condition is not
    satisfied.
    You can check conditions with this function.

    Args: bool_exprs (tuple of Bool expressions): Bool expressions you want to
              evaluate.
    """
    for expr in bool_exprs:
        assert isinstance(expr, Bool)
        expr.expect()
