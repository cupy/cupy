import sys
import unittest

import numpy

from chainer import testing
from chainer.utils import type_check as T


class TestConstant(unittest.TestCase):

    def setUp(self):
        self.x = T.Constant(10)

    def test_str(self):
        self.assertEqual('10', str(self.x))

    def test_eval(self):
        self.assertEqual(10, self.x.eval())


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.x = T.Variable(10, 'x')

    def test_str(self):
        self.assertEqual('x', str(self.x))

    def test_eval(self):
        self.assertEqual(10, self.x.eval())


class Object(object):

    def __init__(self):
        self.value = 10


class TestGetAttr(unittest.TestCase):

    def setUp(self):
        x = Object()
        self.value = T.GetAttr(T.Variable(x, 'x'), 'value')
        self.value2 = T.GetAttr(T.Variable(x, 'x'), T.Constant('value'))
        self.value3 = T.GetAttr(T.Variable(x, 'x'), 3)

    def test_str(self):
        self.assertEqual('x.value', str(self.value))
        self.assertEqual('x.value', str(self.value2))
        self.assertEqual('getattr(x, 3)', str(self.value3))

    def test_eval(self):
        self.assertEqual(10, self.value.eval())


class TestGetItem(unittest.TestCase):

    def setUp(self):
        x = T.Variable([1, 2, 3], 'x')
        y = T.Variable({'a': 1, 'b': 2}, 'y')
        self.x = x
        self.v1 = T.GetItem(x, 1)
        self.v2 = T.GetItem(y, 'a')

    def test_str(self):
        self.assertEqual('x[1]', str(self.v1))
        self.assertEqual("y['a']", str(self.v2))

        x = self.x
        self.assertEqual('x[:]', str(x[:]))
        self.assertEqual('x[:]', str(x[::]))
        self.assertEqual('x[1:]', str(x[1:]))
        self.assertEqual('x[:2]', str(x[:2]))
        self.assertEqual('x[1:2]', str(x[1:2]))
        self.assertEqual('x[1::1]', str(x[1::1]))
        self.assertEqual('x[:2:1]', str(x[:2:1]))
        self.assertEqual('x[1:2:1]', str(x[1:2:1]))
        self.assertEqual('x[...]', str(x[...]))
        self.assertEqual('x[0, 1]', str(x[0, 1]))
        self.assertEqual('x[1:2, ...]', str(x[1:2:, ...]))

    def test_eval(self):
        self.assertEqual(2, self.v1.eval())
        self.assertEqual(1, self.v2.eval())


class TestCall(unittest.TestCase):

    def setUp(self):
        f = T.Variable(sum, 'sum')
        self.c1 = T.Call(f, ([1, 2, 3],))
        self.c2 = f([1, 2, 3])
        self.c3 = T.Call(f, (['', 1],))

    def test_str(self):
        self.assertEqual('sum([1, 2, 3])', str(self.c1))
        self.assertEqual('sum([1, 2, 3])', str(self.c2))
        self.assertEqual('sum([\'\', 1])', str(self.c3))

    def test_eval(self):
        self.assertEqual(6, self.c1.eval())
        self.assertEqual(6, self.c2.eval())
        # an error is occured in `eval`
        with self.assertRaises(TypeError):
            self.assertEqual(6, self.c3.eval())


class TestBinaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.Variable(1, 'x')
        y = T.Variable(1, 'y')

        def f(x, y):
            return x, y
        self.op1 = T.BinaryOperator(7, x, y, '+', f)
        self.op2 = T.BinaryOperator(8, x, y, '+', f)
        self.op3 = T.BinaryOperator(9, x, y, '+', f)

        self.op4 = T.BinaryOperator(7, x, y, '+', f, True)
        self.op5 = T.BinaryOperator(8, x, y, '+', f, True)
        self.op6 = T.BinaryOperator(9, x, y, '+', f, True)

    def test_str(self):
        self.assertEqual('x + y', str(self.op1))
        self.assertEqual('x + (y)', str(self.op2))
        self.assertEqual('(x) + (y)', str(self.op3))

        self.assertEqual('x + y', str(self.op4))
        self.assertEqual('(x) + y', str(self.op5))
        self.assertEqual('(x) + (y)', str(self.op6))

    def test_eval(self):
        self.assertEqual((1, 1), self.op1.eval())


class TestUnaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.Variable(1, 'x')

        def f(x):
            return x,
        self.op1 = T.UnaryOperator(8, x, '-', f)
        self.op2 = T.UnaryOperator(9, x, '-', f)

    def test_str(self):
        self.assertEqual('-x', str(self.op1))
        self.assertEqual('-(x)', str(self.op2))

    def test_eval(self):
        self.assertEqual((1, ), self.op1.eval())


class TestOperators(unittest.TestCase):

    def setUp(self):
        self.x = T.Variable(1, 'x')
        self.y = T.Variable(1, 'y')

    def test_str(self):
        x = self.x
        y = self.y
        self.assertEqual('x + y', str(x + y))
        self.assertEqual('1 + x', str(1 + x))
        self.assertEqual('x - y', str(x - y))
        self.assertEqual('1 - x', str(1 - x))
        self.assertEqual('x * y', str(x * y))
        self.assertEqual('1 * x', str(1 * x))
        self.assertEqual('x / y', str(x / y))
        self.assertEqual('1 / x', str(1 / x))
        self.assertEqual('x // y', str(x // y))
        self.assertEqual('1 // x', str(1 // x))
        self.assertEqual('x % y', str(x % y))
        self.assertEqual('1 % x', str(1 % x))
        self.assertEqual('x ** y', str(x ** y))
        self.assertEqual('x ** y', str(pow(x, y)))
        self.assertEqual('x << y', str(x << y))
        self.assertEqual('1 << x', str(1 << x))
        self.assertEqual('x >> y', str(x >> y))
        self.assertEqual('1 >> x', str(1 >> x))
        self.assertEqual('x & y', str(x & y))
        self.assertEqual('1 & x', str(1 & x))
        self.assertEqual('x ^ y', str(x ^ y))
        self.assertEqual('1 ^ x', str(1 ^ x))
        self.assertEqual('x | y', str(x | y))
        self.assertEqual('1 | x', str(1 | x))

        self.assertEqual('-x', str(-x))
        self.assertEqual('+x', str(+x))
        self.assertEqual('~x', str(~x))

        # left-associative
        self.assertEqual('x + x - x', str(x + x - x))
        self.assertEqual('x + (x - x)', str(x + (x - x)))
        self.assertEqual('x << (x << x)', str(x << (x << x)))

        # right-associative
        self.assertEqual('x ** x ** x', str(x ** x ** x))
        self.assertEqual('x ** x ** x', str(x ** (x ** x)))
        self.assertEqual('(x ** x) ** x', str((x ** x) ** x))

        self.assertEqual('-(x + x)', str(-(x + x)))

        # pow has higher priority than unary operators
        self.assertEqual('-x ** x', str(-x ** x))
        self.assertEqual('(-x) ** x', str((-x) ** x))

    def test_priority(self):
        x = self.x
        y = self.y

        self.assertTrue((x << y).priority == (x >> y).priority)
        self.assertTrue((x + y).priority == (x - y).priority)
        self.assertTrue((x * y).priority ==
                        (x / y).priority ==
                        (x // y).priority ==
                        (x % y).priority)
        self.assertTrue((-x).priority == (+x).priority == (~x).priority)

        self.assertTrue((x | y).priority <
                        (x ^ y).priority <
                        (x & y).priority <
                        (x << y).priority <
                        (x + y).priority <
                        (x * y).priority <
                        (-x).priority <
                        (x ** y).priority <
                        x.priority)


class TestDivOperator(unittest.TestCase):

    def setUp(self):
        self.x = T.Variable(1, 'x')
        self.y = T.Variable(2, 'y')

    def test_div(self):
        # Behavior of '/' operator for int depends on the version of Python
        if sys.version_info < (3, 0, 0):
            self.assertEqual(0, (self.x / self.y).eval())
        else:
            self.assertEqual(0.5, (self.x / self.y).eval())


class TestGetType(unittest.TestCase):

    def test_empty(self):
        ts = T.get_types((), 'name', False)
        self.assertIsInstance(ts, T.TypeInfoTuple)
        self.assertEqual(0, len(ts))
        self.assertEqual('name', ts.name)

    def test_simple(self):
        data = (numpy.zeros((1, 2, 3)).astype(numpy.float32),)
        ts = T.get_types(data, 'name', False)
        self.assertIsInstance(ts, T.TypeInfoTuple)
        self.assertEqual(1, len(ts))
        self.assertEqual('name', ts.name)

        t = ts[0]
        self.assertIsInstance(t, T.Expr)
        self.assertEqual(1, t.shape[0].eval())
        self.assertEqual(2, t.shape[1].eval())
        self.assertEqual(3, t.shape[2].eval())
        self.assertEqual(3, t.ndim.eval())
        self.assertEqual(numpy.float32, t.dtype.eval())

    def test_invalid_arg(self):
        with self.assertRaises(AssertionError):
            T.get_types(1, 'name', False)


class TestBoolBinaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.Variable(1, 'x')
        y = T.Variable(1, 'y')
        z = T.Variable(2, 'z')

        def f(x, y):
            return x == y
        self.op1 = T.BoolBinaryOperator(x, y, '==', '!=', f)
        self.op2 = T.BoolBinaryOperator(x, z, '==', '!=', f)

    def test_eval(self):
        self.assertTrue(self.op1.eval())

    def test_expect(self):
        with self.assertRaises(T.InvalidType):
            self.op2.expect()

    def test_bool(self):
        with self.assertRaises(RuntimeError):
            bool(self.op1)

    def test_bool_operator(self):
        with self.assertRaises(RuntimeError):
            not self.op1


class TestLazyGetItem(unittest.TestCase):

    def setUp(self):
        self.t = T.Constant(0)

    def test_evaluate_size(self):
        # __getitem__, __getattr__ and __call__ only make syntax trees, but
        # they are not evalated yet
        self.assertIsInstance(self.t[1], T.Expr)
        self.assertIsInstance(self.t.x, T.Expr)
        self.assertIsInstance(self.t(), T.Expr)

        # an error is raised on evaluation time
        with self.assertRaises(TypeError):
            self.t[1].eval()
        with self.assertRaises(AttributeError):
            self.t.x.eval()
        with self.assertRaises(TypeError):
            self.t().eval()


class TestListItem(unittest.TestCase):

    def test_eval_list_items(self):
        self.assertTrue((T.Constant([0]) == [T.Constant(0)]).eval())

    def test_list_str(self):
        self.assertEqual('[0]', T._repr([T.Constant(0)]))

    def test_eval_tuple_items(self):
        self.assertTrue((T.Constant((0,)) == (T.Constant(0),)).eval())

    def test_tuple_str(self):
        self.assertEqual('()', T._repr(()))
        self.assertEqual('(0,)', T._repr((T.Constant(0),)))
        self.assertEqual('(0, 0)', T._repr((T.Constant(0), T.Constant(0))))

    def test_eval_nest_list(self):
        self.assertTrue((T.Constant([[0]]) == [[T.Constant(0)]]).eval())

    def test_nest_list_str(self):
        self.assertEqual('[[0]]', T._repr([[T.Constant(0)]]))


class TestProd(unittest.TestCase):

    def test_name(self):
        self.assertEqual(T.prod.name, 'prod')

    def test_value(self):
        self.assertIs(T.prod.value, numpy.prod)


testing.run_module(__name__, __file__)
