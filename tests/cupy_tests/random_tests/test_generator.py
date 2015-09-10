import mock
import unittest

import numpy

from cupy import cuda
from cupy.cuda import curand
from cupy.random import generator
from cupy import testing


@testing.gpu
class TestRandomState(unittest.TestCase):

    _multiprocess_can_split_ = True
    args = (0.0, 1.0)
    size = None

    def setUp(self):
        self.rs = generator.RandomState()

    def check_lognormal(self, curand_func, dtype):
        out = self.rs.lognormal(self.args[0], self.args[1], self.size, dtype)
        curand_func.assert_called_once_with(
            self.rs._generator, out.data.ptr,
            out.size, self.args[0], self.args[1])

    def test_lognormal_float32(self):
        curand.generateLogNormal = mock.Mock()
        self.check_lognormal(curand.generateLogNormal, numpy.float32)

    def test_lognormal_float64(self):
        curand.generateLogNormalDouble = mock.Mock()
        self.check_lognormal(curand.generateLogNormalDouble, numpy.float64)

    def check_normal(self, curand_func, dtype):
        out = self.rs.normal(self.args[0], self.args[1], self.size, dtype)
        curand_func.assert_called_once_with(
            self.rs._generator, out.data.ptr,
            out.size, self.args[0], self.args[1])

    def test_normal_float32(self):
        curand.generateNormal = mock.Mock()
        self.check_normal(curand.generateNormal, numpy.float32)

    def test_normal_float64(self):
        curand.generateNormalDouble = mock.Mock()
        self.check_normal(curand.generateNormalDouble, numpy.float64)

    def check_random_sample(self, curand_func, dtype):
        out = self.rs.random_sample(self.size, dtype)
        curand_func.assert_called_once_with(
            self.rs._generator, out.data.ptr, out.size)

    def test_random_sample_float32(self):
        curand.generateUniform = mock.Mock()
        self.check_random_sample(curand.generateUniform, numpy.float32)

    def test_random_sample_float64(self):
        curand.generateUniformDouble = mock.Mock()
        self.check_random_sample(curand.generateUniformDouble, numpy.float64)

    def check_seed(self, curand_func, seed):
        self.rs.seed(seed)
        call_args_list = curand_func.call_args_list
        self.assertEqual(1, len(call_args_list))
        call_args = call_args_list[0][0]
        self.assertEqual(2, len(call_args))
        self.assertEqual(self.rs._generator, call_args[0])
        self.assertEqual(numpy.uint64, call_args[1].dtype)

    def test_seed_none(self):
        curand.setPseudoRandomGeneratorSeed = mock.Mock()
        self.check_seed(curand.setPseudoRandomGeneratorSeed, None)

    @testing.for_all_dtypes()
    def test_seed_not_none(self, dtype):
        curand.setPseudoRandomGeneratorSeed = mock.Mock()
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
class TestRandAndRandN(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState()

    def test_rand(self):
        generator.random_sample = mock.Mock()
        self.rs.rand(1, 2, 3, dtype=numpy.float32)
        generator.random_sample.assert_call_once_with((1, 2, 3), numpy.float32)

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn(self):
        generator.normal = mock.Mock()
        self.rs.randn(1, 2, 3, dtype=numpy.float32)
        generator.normal.assert_call_once_with((1, 2, 3), numpy.float32)

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.randn(1, 2, 3, unnecessary='unnecessary_argument')


class TestResetStates(unittest.TestCase):

    def test_reset_states(self):
        generator._random_states = 'dummy'
        generator.reset_states()
        self.assertEqual({}, generator._random_states)


@testing.gpu
class TestGetRandomState(unittest.TestCase):

    def setUp(self):
        self.device_id = cuda.Device().id

    def test_get_random_state_initialize(self):
        generator._random_states = {}
        rs = generator.get_random_state()
        self.assertEqual(generator._random_states[self.device_id], rs)

    def test_get_random_state_memoized(self):
        generator._random_states = {self.device_id: 'expected',
                                    self.device_id+1: 'dummy'}
        rs = generator.get_random_state()
        self.assertEqual('expected', generator._random_states[self.device_id])
        self.assertEqual('dummy', generator._random_states[self.device_id+1])
        self.assertEqual('expected', rs)


class TestGetSize(unittest.TestCase):

    def test_none(self):
        self.assertEqual(generator._get_size(None), ())

    def check_collection(self, a):
        self.assertEqual(generator._get_size(a), tuple(a))

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        self.assertEqual(generator._get_size(1), (1,))

    def test_float(self):
        with self.assertRaises(ValueError):
            generator._get_size(1.0)


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
