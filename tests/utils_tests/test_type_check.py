import unittest

import numpy

from chainer.utils import type_check as T


class TestIntConstant(unittest.TestCase):

    def setUp(self):
        self.x = T.IntConstant(10)

    def test_str(self):
        self.assertEqual('10', str(self.x))

    def test_eval(self):
        self.assertEqual(10, self.x.eval())


class TestIntVariable(unittest.TestCase):

    def setUp(self):
        self.x = T.IntVariable(10, 'x')

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
        x = T.IntVariable(1, 'x')
        y = T.IntVariable(1, 'y')
        f = lambda x, y: (x, y)
        self.op1 = T.IntBinaryOperator(7, x, y, '+', f)
        self.op2 = T.IntBinaryOperator(8, x, y, '+', f)
        self.op3 = T.IntBinaryOperator(9, x, y, '+', f)

    def test_str(self):
        self.assertEqual('x + y', str(self.op1))
        self.assertEqual('x + (y)', str(self.op2))
        self.assertEqual('(x) + (y)', str(self.op3))

    def test_eval(self):
        self.assertEqual((1, 1), self.op1.eval())


class TestUnaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.IntVariable(1, 'x')
        f = lambda x: (x,)
        self.op1 = T.IntUnaryOperator(8, x, '-', f)
        self.op2 = T.IntUnaryOperator(9, x, '-', f)

    def test_str(self):
        self.assertEqual('-x', str(self.op1))
        self.assertEqual('-(x)', str(self.op2))

    def test_eval(self):
        self.assertEqual((1, ), self.op1.eval())


class TestOperators(unittest.TestCase):

    def setUp(self):
        self.x = T.IntVariable(1, 'x')
        self.y = T.IntVariable(1, 'y')

    def test_str(self):
        x = self.x
        y = self.y
        self.assertEqual('x + y', str(x + y))
        self.assertEqual('1 + x', str(1 + x))
        self.assertEqual('x - y', str(x - y))
        self.assertEqual('1 - x', str(1 - x))
        self.assertEqual('x * y', str(x * y))
        self.assertEqual('1 * x', str(1 * x))
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

        # left-assosiative
        self.assertEqual('x + x - x', str(x + x - x))
        self.assertEqual('x + (x - x)', str(x + (x - x)))
        self.assertEqual('x << (x << x)', str(x << (x << x)))

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


class TestGetType(unittest.TestCase):

    def test_empty(self):
        ts = T.get_types((), False)
        self.assertIsInstance(ts, T.TypeInfoTuple)
        self.assertEqual(0, len(ts))

    def test_simple(self):
        data = (numpy.zeros((1, 2, 3)).astype(numpy.float32),)
        ts = T.get_types(data, False)
        self.assertIsInstance(ts, T.TypeInfoTuple)
        self.assertEqual(1, len(ts))

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
            T.get_types(1, False)


class TestBoolBinaryOperator(unittest.TestCase):

    def setUp(self):
        x = T.IntVariable(1, 'x')
        y = T.IntVariable(1, 'y')
        z = T.IntVariable(2, 'z')
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
