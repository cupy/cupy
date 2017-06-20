import mock
import unittest

import numpy
import six

from cupy import cuda
from cupy import random
from cupy import testing
from cupy.testing import condition
from cupy.testing import hypothesis


@testing.gpu
class TestRandint(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.rs_tmp = random.generator._random_states
        device_id = cuda.Device().id
        self.m = mock.Mock()
        self.m.interval.return_value = 0
        random.generator._random_states = {device_id: self.m}

    def tearDown(self):
        random.generator._random_states = self.rs_tmp

    def test_value_error(self):
        with self.assertRaises(ValueError):
            random.randint(100, 1)

    def test_high_and_size_are_none(self):
        random.randint(3)
        self.m.interval.assert_called_with(2, None)

    def test_size_is_none(self):
        random.randint(3, 5)
        self.m.interval.assert_called_with(1, None)

    def test_high_is_none(self):
        random.randint(3, None, (1, 2, 3))
        self.m.interval.assert_called_with(2, (1, 2, 3))

    def test_no_none(self):
        random.randint(3, 5, (1, 2, 3))
        self.m.interval.assert_called_with(1, (1, 2, 3))


@testing.fixed_random()
@testing.gpu
class TestRandint2(unittest.TestCase):

    @condition.repeat(3, 10)
    def test_bound_1(self):
        vals = [random.randint(0, 10, (2, 3)).get() for _ in range(10)]
        for val in vals:
            self.assertEqual(val.shape, (2, 3))
        self.assertEqual(min(_.min() for _ in vals), 0)
        self.assertEqual(max(_.max() for _ in vals), 9)

    @condition.repeat(3, 10)
    def test_bound_2(self):
        vals = [random.randint(0, 2).get() for _ in range(20)]
        for val in vals:
            self.assertEqual(val.shape, ())
        self.assertEqual(min(vals), 0)
        self.assertEqual(max(vals), 1)

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        mx = 5
        trial = 100
        vals = [random.randint(mx).get() for _ in six.moves.xrange(trial)]
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 1))[0]
        expected = numpy.array([float(trial) / mx] * mx)
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3, 10)
    def test_goodness_of_fit_2(self):
        mx = 5
        vals = random.randint(mx, size=(5, 20)).get()
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 1))[0]
        expected = numpy.array([float(vals.size) / mx] * mx)
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.gpu
class TestRandomIntegers(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.randint_tmp = random.sample_.randint
        random.sample_.randint = mock.Mock()

    def tearDown(self):
        random.sample_.randint = self.randint_tmp

    def test_normal(self):
        random.random_integers(3, 5)
        random.sample_.randint.assert_called_with(3, 6, None)

    def test_high_is_none(self):
        random.random_integers(3, None)
        random.sample_.randint.assert_called_with(1, 4, None)

    def test_size_is_not_none(self):
        random.random_integers(3, 5, (1, 2, 3))
        random.sample_.randint.assert_called_with(3, 6, (1, 2, 3))


@testing.fixed_random()
@testing.gpu
class TestRandomIntegers2(unittest.TestCase):

    _multiprocess_can_split_ = True

    @condition.repeat(3, 10)
    def test_bound_1(self):
        vals = [random.random_integers(0, 10, (2, 3)).get() for _ in range(10)]
        for val in vals:
            self.assertEqual(val.shape, (2, 3))
        self.assertEqual(min(_.min() for _ in vals), 0)
        self.assertEqual(max(_.max() for _ in vals), 10)

    @condition.repeat(3, 10)
    def test_bound_2(self):
        vals = [random.random_integers(0, 2).get() for _ in range(20)]
        for val in vals:
            self.assertEqual(val.shape, ())
        self.assertEqual(min(vals), 0)
        self.assertEqual(max(vals), 2)

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        mx = 5
        trial = 100
        vals = [random.randint(0, mx).get() for _ in six.moves.xrange(trial)]
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 1))[0]
        expected = numpy.array([float(trial) / mx] * mx)
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3, 10)
    def test_goodness_of_fit_2(self):
        mx = 5
        vals = random.randint(0, mx, (5, 20)).get()
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 1))[0]
        expected = numpy.array([float(vals.size) / mx] * mx)
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.gpu
class TestChoice(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.rs_tmp = random.generator._random_states
        device_id = cuda.Device().id
        self.m = mock.Mock()
        self.m.choice.return_value = 0
        random.generator._random_states = {device_id: self.m}

    def tearDown(self):
        random.generator._random_states = self.rs_tmp

    def test_size_and_replace_and_p_are_none(self):
        random.choice(3)
        self.m.choice.assert_called_with(3, None, True, None)

    def test_size_and_replace_are_none(self):
        random.choice(3, None, None, [0.1, 0.1, 0.8])
        self.m.choice.assert_called_with(3, None, None, [0.1, 0.1, 0.8])

    def test_size_and_p_are_none(self):
        random.choice(3, None, True)
        self.m.choice.assert_called_with(3, None, True, None)

    def test_replace_and_p_are_none(self):
        random.choice(3, 1)
        self.m.choice.assert_called_with(3, 1, True, None)

    def test_size_is_none(self):
        random.choice(3, None, True, [0.1, 0.1, 0.8])
        self.m.choice.assert_called_with(3, None, True, [0.1, 0.1, 0.8])

    def test_replace_is_none(self):
        random.choice(3, 1, None, [0.1, 0.1, 0.8])
        self.m.choice.assert_called_with(3, 1, None, [0.1, 0.1, 0.8])

    def test_p_is_none(self):
        random.choice(3, 1, True)
        self.m.choice.assert_called_with(3, 1, True, None)

    def test_no_none(self):
        random.choice(3, 1, True, [0.1, 0.1, 0.8])
        self.m.choice.assert_called_with(3, 1, True, [0.1, 0.1, 0.8])


@testing.gpu
class TestRandomSample(unittest.TestCase):

    def test_rand(self):
        random.sample_.random_sample = mock.Mock()
        random.rand(1, 2, 3, dtype=numpy.float32)
        random.sample_.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_rand_default_dtype(self):
        random.sample_.random_sample = mock.Mock()
        random.rand(1, 2, 3)
        random.sample_.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=float)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        random.distributions.normal = mock.Mock()
        random.randn(1, 2, 3, dtype=numpy.float32)
        random.distributions.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_randn_default_dtype(self):
        random.distributions.normal = mock.Mock()
        random.randn(1, 2, 3)
        random.distributions.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=float)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            random.randn(1, 2, 3, unnecessary='unnecessary_argument')


@testing.parameterize(
    {'size': None},
    {'size': ()},
    {'size': 4},
    {'size': (0,)},
    {'size': (1, 0)},
)
@testing.fixed_random()
@testing.gpu
class TestMultinomial(unittest.TestCase):

    _multiprocess_can_split_ = True

    @condition.repeat(3, 10)
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(rtol=0.05)
    def test_multinomial(self, xp, dtype):
        pvals = xp.array([0.2, 0.3, 0.5], dtype)
        x = xp.random.multinomial(100000, pvals, self.size)
        self.assertEqual(x.dtype, 'l')
        return x / 100000
