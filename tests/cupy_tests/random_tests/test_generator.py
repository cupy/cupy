import mock
import operator
import os
import unittest

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy.cuda import curand
from cupy.random import generator
from cupy import testing
from cupy.testing import condition
from cupy.testing import hypothesis


class FunctionSwitcher(object):

    def __init__(self, f):
        self.tmp = f
        self.func_name = f.__name__

    def __enter__(self):
        setattr(curand, self.func_name, mock.Mock())

    def __exit__(self, *_):
        setattr(curand, self.func_name, self.tmp)


@testing.gpu
class TestRandomState(unittest.TestCase):

    _multiprocess_can_split_ = True
    args = (0.0, 1.0)
    size = None

    def setUp(self):
        self.rs = generator.RandomState()

    def check_lognormal(self, curand_func, dtype):
        shape = core.get_size(self.size)
        exp_size = six.moves.reduce(operator.mul, shape, 1)
        if exp_size % 2 == 1:
            exp_size += 1

        curand_func.return_value = cupy.zeros(exp_size, dtype=dtype)
        out = self.rs.lognormal(self.args[0], self.args[1], self.size, dtype)
        gen, _, size, mean, sigma = curand_func.call_args[0]
        self.assertIs(gen, self.rs._generator)
        self.assertEqual(size, exp_size)
        self.assertIs(mean, self.args[0])
        self.assertIs(sigma, self.args[1])
        self.assertEqual(out.shape, shape)

    def test_lognormal_float(self):
        with FunctionSwitcher(curand.generateLogNormalDouble):
            self.check_lognormal(curand.generateLogNormalDouble, float)

    def test_lognormal_float32(self):
        with FunctionSwitcher(curand.generateLogNormal):
            self.check_lognormal(curand.generateLogNormal, numpy.float32)

    def test_lognormal_float64(self):
        with FunctionSwitcher(curand.generateLogNormalDouble):
            self.check_lognormal(curand.generateLogNormalDouble, numpy.float64)

    def check_normal(self, curand_func, dtype):
        shape = core.get_size(self.size)
        exp_size = six.moves.reduce(operator.mul, shape, 1)
        if exp_size % 2 == 1:
            exp_size += 1

        curand_func.return_value = cupy.zeros(exp_size, dtype=dtype)
        out = self.rs.normal(self.args[0], self.args[1], self.size, dtype)
        gen, _, size, loc, scale = curand_func.call_args[0]
        self.assertIs(gen, self.rs._generator)
        self.assertEqual(size, exp_size)
        self.assertIs(loc, self.args[0])
        self.assertIs(scale, self.args[1])
        self.assertEqual(out.shape, shape)

    def test_normal_float32(self):
        with FunctionSwitcher(curand.generateNormal):
            self.check_normal(curand.generateNormal, numpy.float32)

    def test_normal_float64(self):
        with FunctionSwitcher(curand.generateNormalDouble):
            self.check_normal(curand.generateNormalDouble, numpy.float64)

    def check_random_sample(self, curand_func, dtype):
        out = self.rs.random_sample(self.size, dtype)
        curand_func.assert_called_once_with(
            self.rs._generator, out.data.ptr, out.size)

    def test_random_sample_float32(self):
        with FunctionSwitcher(curand.generateUniform):
            self.check_random_sample(curand.generateUniform, numpy.float32)

    def test_random_sample_float64(self):
        with FunctionSwitcher(curand.generateUniformDouble):
            self.check_random_sample(
                curand.generateUniformDouble, numpy.float64)

    def check_seed(self, curand_func, seed):
        self.rs.seed(seed)
        call_args_list = curand_func.call_args_list
        self.assertEqual(1, len(call_args_list))
        call_args = call_args_list[0][0]
        self.assertEqual(2, len(call_args))
        self.assertIs(self.rs._generator, call_args[0])
        self.assertEqual(numpy.uint64, call_args[1].dtype)

    def test_seed_none(self):
        with FunctionSwitcher(curand.setPseudoRandomGeneratorSeed):
            self.check_seed(curand.setPseudoRandomGeneratorSeed, None)

    @testing.for_all_dtypes()
    def test_seed_not_none(self, dtype):
        with FunctionSwitcher(curand.setPseudoRandomGeneratorSeed):
            self.check_seed(curand.setPseudoRandomGeneratorSeed, dtype(0))


@testing.gpu
class TestRandomState2(TestRandomState):

    args = (10.0, 20.0)
    size = None


@testing.gpu
class TestRandomState3(TestRandomState):

    args = (0.0, 1.0)
    size = 10


@testing.gpu
class TestRandomState4(TestRandomState):

    args = (0.0, 1.0)
    size = (1, 2, 3)


@testing.gpu
class TestRandomState6(TestRandomState):

    args = (0.0, 1.0)
    size = 3


@testing.gpu
class TestRandomState7(TestRandomState):

    args = (0.0, 1.0)
    size = (3, 3)


@testing.gpu
class TestRandomState8(TestRandomState):

    args = (0.0, 1.0)
    size = ()


@testing.gpu
class TestRandAndRandN(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState()

    def test_rand(self):
        self.rs.random_sample = mock.Mock()
        self.rs.rand(1, 2, 3, dtype=numpy.float32)
        self.rs.random_sample.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        self.rs.normal = mock.Mock()
        self.rs.randn(1, 2, 3, dtype=numpy.float32)
        self.rs.normal.assert_called_once_with(
            size=(1, 2, 3), dtype=numpy.float32)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.randn(1, 2, 3, unnecessary='unnecessary_argument')


@testing.gpu
class TestInterval(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState()

    def test_zero(self):
        numpy.testing.assert_array_equal(
            self.rs.interval(0, (2, 3)).get(), numpy.zeros((2, 3)))

    def test_shape_zero(self):
        v = self.rs.interval(10, None)
        self.assertEqual(v.dtype, 'i')
        self.assertEqual(v.shape, ())

    def test_shape_one_dim(self):
        v = self.rs.interval(10, 10)
        self.assertEqual(v.dtype, 'i')
        self.assertEqual(v.shape, (10,))

    def test_shape_multi_dim(self):
        v = self.rs.interval(10, (1, 2))
        self.assertEqual(v.dtype, 'i')
        self.assertEqual(v.shape, (1, 2))

    @condition.repeat(10)
    def test_within_interval(self):
        val = self.rs.interval(10, (2, 3)).get()
        numpy.testing.assert_array_less(
            numpy.full((2, 3), -1, dtype=numpy.int64), val)
        numpy.testing.assert_array_less(
            val, numpy.full((2, 3), 11, dtype=numpy.int64))

    @condition.retry(20)
    def test_lower_bound(self):
        val = self.rs.interval(2, None).get()
        self.assertEqual(0, val)

    @condition.retry(20)
    def test_upper_bound(self):
        val = self.rs.interval(2, None).get()
        self.assertEqual(2, val)

    @condition.retry(5)
    def test_goodness_of_fit(self):
        mx = 5
        trial = 100
        vals = [self.rs.interval(mx, None).get()
                for _ in six.moves.xrange(trial)]
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(trial) / mx + 1] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.retry(5)
    def test_goodness_of_fit_2(self):
        mx = 5
        vals = self.rs.interval(mx, (5, 5)).get()
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(vals.size) / mx + 1] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


class TestResetStates(unittest.TestCase):

    def test_reset_states(self):
        generator._random_states = 'dummy'
        generator.reset_states()
        self.assertEqual({}, generator._random_states)


@testing.gpu
class TestGetRandomState(unittest.TestCase):

    def setUp(self):
        self.device_id = cuda.Device().id
        self.rs_tmp = generator._random_states

    def tearDown(self, *args):
        generator._random_states = self.rs_tmp

    def test_get_random_state_initialize(self):
        generator._random_states = {}
        rs = generator.get_random_state()
        self.assertEqual(generator._random_states[self.device_id], rs)

    def test_get_random_state_memoized(self):
        generator._random_states = {self.device_id: 'expected',
                                    self.device_id + 1: 'dummy'}
        rs = generator.get_random_state()
        self.assertEqual('expected', generator._random_states[self.device_id])
        self.assertEqual('dummy', generator._random_states[self.device_id + 1])
        self.assertEqual('expected', rs)


@testing.gpu
class TestGetRandomState2(unittest.TestCase):

    def setUp(self):
        self.rs_tmp = generator.RandomState
        generator.RandomState = mock.Mock()
        self.rs_dict = generator._random_states
        generator._random_states = {}
        self.chainer_seed = os.getenv('CHAINER_SEED')

    def tearDown(self, *args):
        generator.RandomState = self.rs_tmp
        generator._random_states = self.rs_dict
        if self.chainer_seed is None:
            os.environ.pop('CHAINER_SEED', None)
        else:
            os.environ['CHAINER_SEED'] = self.chainer_seed

    def test_get_random_state_no_chainer_seed(self):
        os.environ.pop('CHAINER_SEED', None)
        generator.get_random_state()
        generator.RandomState.assert_called_with(None)

    def test_get_random_state_with_chainer_seed(self):
        os.environ['CHAINER_SEED'] = '1'
        generator.get_random_state()
        generator.RandomState.assert_called_with('1')


class TestCheckAndGetDtype(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def test_float32_64_type(self, dtype):
        self.assertEqual(generator._check_and_get_dtype(dtype),
                         numpy.dtype(dtype))

    def test_float16(self):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(numpy.float16)

    @testing.for_int_dtypes()
    def test_int_type(self, dtype):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(dtype)
