import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


def _full_like(x, val):
    xp = cuda.get_array_module(x)
    return xp.full_like(x, val)


def _zeros_like(x):
    xp = cuda.get_array_module(x)
    return xp.zeros_like(x)


def _dot(x, y):
    return sum(map(lambda a: a[0] * a[1], zip(x, y)))


class NumericalGradientTest(unittest.TestCase):

    eps = None

    def f(self, xs):
        return xs[0] ** 2,

    def df(self, xs):
        return (2 * xs[0],),

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (_uniform(2, 1),)

    def check_numerical_grad_one(self, f, df, xs, gys, eps):
        dfxs = df(xs)

        gys = tuple(0 if gy is None else gy for gy in gys)
        # matrix-vector multiplication of dfxs and dys
        dx_expect = tuple(map(lambda dfx: _dot(dfx, gys), dfxs))

        def func():
            return f(xs)
        dx_actual = gradient_check.numerical_grad(func, xs, gys, eps)

        print('eps: {}'.format(eps))
        self.assertEqual(len(dx_expect), len(dx_actual))
        for e, a in zip(dx_expect, dx_actual):
            testing.assert_allclose(e, a, atol=1e-3, rtol=1e-3)

    def check_numerical_grad(self, f, df, xs, gys, eps=None):
        if eps is None:
            eps = tuple(10 ** (-i) for i in six.moves.range(2, 5))
        elif not isinstance(eps, tuple):
            eps = (eps, )

        for e in eps:
            self.check_numerical_grad_one(f, df, xs, gys, e)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_numerical_grad(self.f, self.df, self.xs, self.gys,
                                  eps=self.eps)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        gys = tuple(None if gy is None else cuda.to_gpu(gy)
                    for gy in self.gys)

        self.check_numerical_grad(self.f, self.df,
                                  tuple(map(cuda.to_gpu, self.xs)), gys,
                                  eps=self.eps)


class NumericalGradientTest2(NumericalGradientTest):

    def f(self, xs):
        return 1,

    def df(self, xs):
        return (0,),


class NumericalGradientTest3(NumericalGradientTest):

    # Too small eps causes cancellation of significant digits
    eps = (1e-2, 1e-3)

    def f(self, xs):
        xp = cuda.get_array_module(*xs)
        return xp.exp(xs[0]),

    def df(self, xs):
        xp = cuda.get_array_module(*xs)
        return (xp.exp(xs[0]),),

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (_uniform(2, 1),)


class NumericalGradientTest4(NumericalGradientTest):

    def f(self, xs):
        assert len(xs) == 2
        return (2 * xs[0] + 3 * xs[1],
                4 * xs[0] + 5 * xs[1],
                6 * xs[0] + 7 * xs[1])

    def df(self, xs):
        assert len(xs) == 2
        return (
            (_full_like(xs[0], 2), _full_like(xs[0], 4), _full_like(xs[0], 6)),
            (_full_like(xs[1], 3), _full_like(xs[1], 5), _full_like(xs[1], 7)))

    def setUp(self):
        self.xs = tuple(_uniform(2, 1) for _ in six.moves.range(2))
        self.gys = tuple(_uniform(2, 1) for _ in six.moves.range(3))


class NumericalGradientTest5(NumericalGradientTest4):

    def f(self, xs):
        assert len(xs) == 2
        return (2 * xs[0] + 3 * xs[1],
                4 * xs[0] + 5 * xs[1],
                6 * xs[0] + 7 * xs[1])

    def df(self, xs):
        assert len(xs) == 2
        return (
            (_full_like(xs[0], 2), _zeros_like(xs[0]), _full_like(xs[0], 6)),
            (_full_like(xs[1], 3), _zeros_like(xs[1]), _full_like(xs[1], 7)))

    def setUp(self):
        super(NumericalGradientTest5, self).setUp()
        self.gys = (_uniform(2, 1), None, _uniform(2, 1))


class NumericalGradientTest6(NumericalGradientTest):

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (None,)


class NumericalGradientReferenceTest(unittest.TestCase):

    def setUp(self):
        self.x = _uniform(2, 3)

    def check_reference(self, x):
        # A returned value and an input refers the same memory.
        # See issue #488
        def func():
            return x,
        gx, = gradient_check.numerical_grad(func, (x,), (1,))
        testing.assert_allclose(cuda.to_cpu(gx), 1)

    def test_reference_cpu(self):
        self.check_reference(self.x)

    @attr.gpu
    def test_reference_gpu(self):
        self.check_reference(cuda.to_gpu(self.x))


class NumericalGradientInvalidEps(NumericalGradientTest):

    def check_invalid_eps(self, xs, gys, eps):
        with self.assertRaises(AssertionError):
            self.check_numerical_grad(self.f, self.df, xs, gys, eps)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_invalid_eps(self.xs, self.gys, 0)
        self.check_invalid_eps(self.xs, self.gys, -1.0)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        xs = tuple(map(cuda.to_gpu, self.xs))
        gys = tuple(None if gy is None else cuda.to_gpu(gy)
                    for gy in self.gys)

        self.check_invalid_eps(xs, gys, 0)
        self.check_invalid_eps(xs, gys, -1.0)


class NumericalGradientInvalidType(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array(0)
        self.y = numpy.array(0)
        self.f = lambda: None

    @attr.gpu
    def test_invalid_inputs(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (self.x, y), ())

    @attr.gpu
    def test_invalid_outputs(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (), (self.x, y))

    @attr.gpu
    def test_invalid_mixed(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (self.x,), (y,))


class NumericalGradientEpsTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array(0.0, dtype=numpy.float32)
        self.y = numpy.array(1.0, dtype=numpy.float32)

    def check_different_eps(self, x, y):
        def f():
            if -1 < x < 1:
                return x.copy(),
            elif -2 < x < 2:
                return 2 * x,
            else:
                return 0,

        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=0.5)
        self.assertEqual(gx, 1.)
        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=1.5)
        self.assertEqual(gx, 2.)
        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=2.5)
        self.assertEqual(gx, 0.)

    def test_differenct_eps_cpu(self):
        self.check_different_eps(self.x, self.y)

    @attr.gpu
    def test_differenct_eps_gpu(self):
        self.check_different_eps(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class AssertAllCloseTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_identical(self, x):
        testing.assert_allclose(x, x, atol=0, rtol=0)

    @condition.repeat(5)
    def test_identical_cpu(self):
        self.check_identical(self.x)

    @condition.repeat(5)
    @attr.gpu
    def test_identical_gpu(self):
        self.check_identical(cuda.to_gpu(self.x))

    def check_atol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_abs_diff = numpy.max(numpy.abs(x_cpu - y_cpu))
        with self.assertRaises(AssertionError):
            testing.assert_allclose(x, y, atol=max_abs_diff - 1, rtol=0)
        testing.assert_allclose(x, y, atol=max_abs_diff + 1, rtol=0)

    @condition.repeat(5)
    def test_atol_cpu(self):
        self.check_atol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_atol_gpu(self):
        self.check_atol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class AssertAllCloseTest2(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(1, 2, (2, 3)).astype(numpy.float32)

    def check_rtol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_ratio = numpy.max(numpy.abs(x_cpu - y_cpu) / y_cpu)
        with self.assertRaises(AssertionError):
            testing.assert_allclose(x, y, atol=0, rtol=max_ratio - 1)
        testing.assert_allclose(x, y, atol=0, rtol=max_ratio + 1)

    @condition.repeat(5)
    def test_rtol_cpu(self):
        self.check_rtol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_rtol_gpu(self):
        self.check_rtol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class Ident(chainer.Function):

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grads):
        return grads


# numpy.float16 is not tested because it is low precision.
@testing.parameterize(*testing.product({
    'dtype': [None, numpy.float32, numpy.float64],
}))
class TestCheckBackward(unittest.TestCase):

    def test_multiple_output(self):
        x1 = numpy.array([1], dtype='f')
        x2 = numpy.array([1], dtype='f')
        g1 = numpy.array([1], dtype='f')
        g2 = numpy.array([1], dtype='f')

        def f(x, y):
            s, t = Ident()(x, y)
            u = Ident()(t)
            return s, u

        gradient_check.check_backward(f, (x1, x2), (g1, g2), dtype=self.dtype)

    def test_no_grads_for_not_float(self):
        x1 = numpy.array([1], dtype='f')
        x2 = numpy.array([0, 1], dtype='i')  # grad check for this is skipped
        g1 = numpy.array([1], dtype='f')

        def f(x, y):
            s = Ident()(x)
            return s,

        gradient_check.check_backward(f, (x1, x2), g1)

    def test_no_grads_option(self):
        x1 = numpy.array([1], dtype='f')
        x2 = numpy.array([1], dtype='f')  # grad check for this is skipped
        g1 = numpy.array([1], dtype='f')

        def f(x, y):
            s = Ident()(x)
            return s,

        self.assertRaises(TypeError, gradient_check.check_backward,
                          f, (x1, x2), g1, no_grads=[False, False])
        gradient_check.check_backward(f, (x1, x2), g1, no_grads=[False, True])


testing.run_module(__name__, __file__)
