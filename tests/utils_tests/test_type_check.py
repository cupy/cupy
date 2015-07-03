import sys
import unittest

import numpy

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


class TestShape(unittest.TestCase):

    def setUp(self):
        self.x = T.Shape(10, 3, 'xxx')

    def test_str(self):
        self.assertEqual('xxx.shape[3]', str(self.x))

    def test_eval(self):
        self.assertEqual(10, self.x.eval())


class TestMember(unittest.TestCase):

    def setUp(self):
        self.x = T.Member(10, 'xxx', 'yyy')

    def test_str(self):
        self.assertEqual('xxx.yyy', str(self.x))

    def test_eval(self):
        self.assertEqual(10, self.x.eval())


class TestBinaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.Variable(1, 'x')
        y = T.Variable(1, 'y')
        f = lambda x, y: (x, y)
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
        f = lambda x: (x,)
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
        self.assertIsInstance(t, T.TypeInfo)
        self.assertEqual(0, t.index)
        self.assertEqual(3, len(t.shape))
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
        f = lambda x, y: x == y
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
