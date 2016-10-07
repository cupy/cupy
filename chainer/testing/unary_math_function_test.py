import numpy
import unittest

import chainer
from chainer import cuda
from chainer.testing import attr
from chainer.testing import condition


def _make_data_default(shape, dtype):
    x = numpy.random.uniform(-1, 1, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


def unary_math_function_unittest(func, func_expected=None, label_expected=None,
                                 make_data=None):
    """Decorator for testing unary mathematical Chainer functions.

    This decorator makes test classes test unary mathematical Chainer
    functions. Tested are forward and backward computations on CPU and GPU
    across parameterized ``shape`` and ``dtype``.

    Args:
        func(~chainer.Function): Chainer function to be tested by the decorated
            test class.
        func_expected: Function used to provide expected values for
            testing forward computation. If not given, a corresponsing numpy
            function for ``func`` is implicitly picked up by its class name.
        label_expected(string): String used to test labels of Chainer
            functions. If not given, the class name of ``func`` lowered is
            implicitly used.
        make_data: Function to customize input and gradient data used
            in the tests. It takes ``shape`` and ``dtype`` as its arguments,
            and returns a tuple of input and gradient data. By default, uniform
            destribution ranged ``[-1, 1]`` is used for both.

    The decorated test class tests forward and backward computations on CPU and
    GPU across the following :func:`~chainer.testing.parameterize` ed
    parameters:

    - shape: rank of zero, and rank of more than zero
    - dtype: ``numpy.float16``, ``numpy.float32`` and ``numpy.float64``

    Additionally, it tests the label of the Chainer function.

    Chainer functions tested by the test class decorated with the decorator
    should have the following properties:

    - Unary, taking one parameter and returning one value
    - ``dtype`` of input and output are the same
    - Elementwise operation for the supplied ndarray

    .. admonition:: Example

       The following code defines a test class that tests
       :func:`~chainer.functions.sin` Chainer function, which takes a parameter
       with ``dtype`` of float and returns a value with the same ``dtype``.

       .. doctest::

          >>> import unittest
          >>> from chainer import testing
          >>> from chainer import functions as F
          >>>
          >>> @testing.unary_math_function_unittest(F.Sin())
          ... class TestSin(unittest.TestCase):
          ...     pass

       Because the test methods are implicitly injected to ``TestSin`` class by
       the decorator, it is enough to place ``pass`` in the class definition.

       Now the test is run with ``nose`` module.

       .. doctest::

          >>> import nose
          >>> nose.run(
          ...     defaultTest=__name__, argv=['', '-a', '!gpu'], exit=False)
          True

       To customize test data, ``make_data`` optional parameter can be used.
       The following is an example of testing ``sqrt`` Chainer function, which
       is tested in positive value domain here instead of the default input.

       .. doctest::

          >>> import numpy
          >>>
          >>> def make_data(shape, dtype):
          ...     x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
          ...     gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
          ...     return x, gy
          ...
          >>> @testing.unary_math_function_unittest(F.Sqrt(),
          ...                                       make_data=make_data)
          ... class TestSqrt(unittest.TestCase):
          ...     pass
          ...
          >>> nose.run(
          ...     defaultTest=__name__, argv=['', '-a', '!gpu'], exit=False)
          True

       ``make_data`` function which returns input and gradient data generated
       in proper value domains with given ``shape`` and ``dtype`` parameters is
       defined, then passed to the decorator's ``make_data`` parameter.

    """

    # TODO(takagi) In the future, the Chainer functions that could be tested
    #     with the decorator would be extended as:
    #
    #     - Multiple input parameters
    #     - Multiple output values
    #     - Other types than float: integer
    #     - Other operators other than analytic math: basic math

    # Import here to avoid mutual import.
    from chainer import gradient_check
    from chainer import testing

    if func_expected is None:
        name = func.__class__.__name__.lower()
        try:
            func_expected = getattr(numpy, name)
        except AttributeError:
            raise ValueError("NumPy has no functions corresponding "
                             "to Chainer function '{}'.".format(name))

    if label_expected is None:
        label_expected = func.__class__.__name__.lower()

    if make_data is None:
        make_data = _make_data_default

    def f(klass):
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            self.x, self.gy = make_data(self.shape, self.dtype)
            if self.dtype == numpy.float16:
                self.backward_options = {
                    'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4,
                    'dtype': numpy.float64}
            else:
                self.backward_options = {'atol': 1e-4, 'rtol': 1e-4}
        setattr(klass, "setUp", setUp)

        def check_forward(self, x_data):
            x = chainer.Variable(x_data)
            y = func(x)
            self.assertEqual(y.data.dtype, x_data.dtype)
            y_expected = func_expected(cuda.to_cpu(x_data), dtype=x_data.dtype)
            testing.assert_allclose(y_expected, y.data, atol=1e-4, rtol=1e-4)
        setattr(klass, "check_forward", check_forward)

        @condition.retry(3)
        def test_forward_cpu(self):
            self.check_forward(self.x)
        setattr(klass, "test_forward_cpu", test_forward_cpu)

        @attr.gpu
        @condition.retry(3)
        def test_forward_gpu(self):
            self.check_forward(cuda.to_gpu(self.x))
        setattr(klass, "test_forward_gpu", test_forward_gpu)

        def check_backward(self, x_data, y_grad):
            gradient_check.check_backward(
                func, x_data, y_grad, **self.backward_options)
        setattr(klass, "check_backward", check_backward)

        @condition.retry(3)
        def test_backward_cpu(self):
            self.check_backward(self.x, self.gy)
        setattr(klass, "test_backward_cpu", test_backward_cpu)

        @attr.gpu
        @condition.retry(3)
        def test_backward_gpu(self):
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
        setattr(klass, "test_backward_gpu", test_backward_gpu)

        def test_label(self):
            self.assertEqual(func.label, label_expected)
        setattr(klass, "test_label", test_label)

        # Return parameterized class.
        return testing.parameterize(*testing.product({
            'shape': [(3, 2), ()],
            'dtype': [numpy.float16, numpy.float32, numpy.float64]
        }))(klass)
    return f
