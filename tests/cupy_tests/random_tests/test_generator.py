import os
import threading
import unittest

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy.random import generator
from cupy import testing
from cupy.testing import condition
from cupy.testing import hypothesis


class RandomGeneratorTestCase(unittest.TestCase):

    target_method = None

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def _get_generator_func(self, *args, **kwargs):
        assert isinstance(self.target_method, str), (
            'generate_method must be overridden')
        f = getattr(self.rs, self.target_method)
        return lambda: f(*args, **kwargs)

    def _generate_check_repro(self, func, seed=0):
        # Sample a random array while checking reproducibility
        self.rs.seed(seed)
        x = func()
        self.rs.seed(seed)
        y = func()
        testing.assert_array_equal(
            x, y,
            'Randomly generated arrays with the same seed did not match')
        return x

    def generate(self, *args, **kwargs):
        # Pick one sample from generator.
        # Reproducibility is checked by repeating seed-and-sample cycle twice.
        func = self._get_generator_func(*args, **kwargs)
        return self._generate_check_repro(func, seed=0)

    def generate_many(self, *args, **kwargs):
        # Pick many samples from generator.
        # Reproducibility is checked only for the first sample,
        # because it's very slow to set seed every time.
        _count = kwargs.pop('_count', None)
        assert _count is not None, '_count is required'
        func = self._get_generator_func(*args, **kwargs)

        if _count == 0:
            return []

        vals = [self._generate_check_repro(func, seed=0)]
        for i in range(1, _count):
            vals.append(func())
        return vals


@testing.fix_random()
@testing.gpu
class TestRandomState(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def check_seed(self, seed):
        rs = self.rs

        rs.seed(seed)
        xs1 = [rs.uniform() for _ in range(100)]

        rs.seed(seed)
        xs2 = [rs.uniform() for _ in range(100)]

        rs.seed(seed)
        rs.seed(None)
        xs3 = [rs.uniform() for _ in range(100)]

        # Random state must be reproducible
        assert xs1 == xs2
        # Random state must be initialized randomly with seed=None
        assert xs1 != xs3

    @testing.for_int_dtypes()
    def test_seed_not_none(self, dtype):
        self.check_seed(dtype(0))

    @testing.for_dtypes([numpy.complex_])
    def test_seed_invalid_type_complex(self, dtype):
        with self.assertRaises(TypeError):
            self.rs.seed(dtype(0))

    @testing.for_float_dtypes()
    def test_seed_invalid_type_float(self, dtype):
        with self.assertRaises(TypeError):
            self.rs.seed(dtype(0))


@testing.parameterize(
    {'a': 1.0, 'b': 3.0},
    {'a': 3.0, 'b': 3.0},
    {'a': 3.0, 'b': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestBeta(RandomGeneratorTestCase):

    target_method = 'beta'

    def test_beta(self):
        self.generate(a=self.a, b=self.b, size=(3, 2))


@testing.parameterize(
    {'n': 5, 'p': 0.5},
    {'n': 5, 'p': 0.0},
    {'n': 5, 'p': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestBinomial(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'binomial'

    def test_binomial(self):
        self.generate(n=self.n, p=self.p, size=(3, 2))


@testing.gpu
@testing.fix_random()
class TestLaplace(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'laplace'

    def test_laplace_1(self):
        self.generate()

    def test_laplace_2(self):
        self.generate(0.0, 1.0, size=(3, 2))


@testing.gpu
@testing.parameterize(*[
    {'args': (0.0, 1.0), 'size': None},
    {'args': (10.0, 20.0), 'size': None},
    {'args': (0.0, 1.0), 'size': 10},
    {'args': (0.0, 1.0), 'size': (1, 2, 3)},
    {'args': (0.0, 1.0), 'size': 3},
    {'args': (0.0, 1.0), 'size': (3, 3)},
    {'args': (0.0, 1.0), 'size': ()},
])
@testing.fix_random()
class TestLogNormal(RandomGeneratorTestCase):

    target_method = 'lognormal'

    def check_lognormal(self, dtype):
        vals = self.generate_many(
            self.args[0], self.args[1], self.size, dtype, _count=10)

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape
            assert (0 <= val).all()
        # TODO(niboshi): Distribution test

    def test_lognormal_float(self):
        self.check_lognormal(float)

    def test_lognormal_float32(self):
        self.check_lognormal(numpy.float32)

    def test_lognormal_float64(self):
        self.check_lognormal(numpy.float64)


@testing.gpu
@testing.parameterize(*[
    {'args': (0.0, 1.0), 'size': None},
    {'args': (10.0, 20.0), 'size': None},
    {'args': (0.0, 1.0), 'size': 10},
    {'args': (0.0, 1.0), 'size': (1, 2, 3)},
    {'args': (0.0, 1.0), 'size': 3},
    {'args': (0.0, 1.0), 'size': (3, 3)},
    {'args': (0.0, 1.0), 'size': ()},
])
@testing.fix_random()
class TestNormal(RandomGeneratorTestCase):

    target_method = 'normal'

    def check_normal(self, dtype):
        vals = self.generate_many(
            self.args[0], self.args[1], self.size, dtype, _count=10)

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape
        # TODO(niboshi): Distribution test

    def test_normal_float32(self):
        self.check_normal(numpy.float32)

    def test_normal_float64(self):
        self.check_normal(numpy.float64)


@testing.gpu
@testing.parameterize(*[
    {'size': None},
    {'size': 10},
    {'size': (1, 2, 3)},
    {'size': 3},
    {'size': ()},
])
@testing.fix_random()
class TestRandomSample(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def check_random_sample(self, dtype):
        vals = [self.rs.random_sample(self.size, dtype) for _ in range(10)]

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape
            assert (0 <= val).all()
            assert (val < 1).all()
        # TODO(niboshi): Distribution test

    def test_random_sample_float32(self):
        self.check_random_sample(numpy.float32)

    def test_random_sample_float64(self):
        self.check_random_sample(numpy.float64)


@testing.fix_random()
@testing.gpu
class TestRandAndRandN(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.randn(1, 2, 3, unnecessary='unnecessary_argument')


@testing.fix_random()
@testing.gpu
class TestInterval(RandomGeneratorTestCase):

    target_method = 'interval'

    def test_zero(self):
        shape = (2, 3)
        vals = self.generate_many(0, shape, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == shape
            assert (val == 0).all()

    def test_shape_zero(self):
        mx = 10
        vals = self.generate_many(mx, None, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == ()
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_shape_one_dim(self):
        mx = 10
        size = 20
        vals = self.generate_many(mx, size, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == (size,)
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_shape_multi_dim(self):
        mx = 10
        shape = (1, 2)
        vals = self.generate_many(mx, shape, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == shape
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_int32_range(self):
        v = self.generate(0x00000000, 2)
        assert v.dtype == numpy.int32

        v = self.generate(0x7fffffff, 2)
        assert v.dtype == numpy.int32

    def test_uint32_range(self):
        v = self.generate(0x80000000, 2)
        assert v.dtype == numpy.uint32

        v = self.generate(0xffffffff, 2)
        assert v.dtype == numpy.uint32

    def test_bound_1(self):
        vals = self.generate_many(10, (2, 3), _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == (2, 3)
            assert (0 <= val).all()
            assert (val <= 10).all()

    def test_bound_2(self):
        vals = self.generate_many(2, None, _count=20)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == numpy.int32
            assert val.shape == ()
            assert (0 <= val).all()
            assert (val <= 2).all()

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        mx = 5
        trial = 100
        vals = self.generate_many(mx, None, _count=trial)
        vals = [val.get() for val in vals]
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(trial) / (mx + 1)] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3)
    def test_goodness_of_fit_2(self):
        mx = 5
        vals = self.generate(mx, (5, 5)).get()
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(vals.size) / (mx + 1)] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.fix_random()
@testing.gpu
class TestTomaxint(RandomGeneratorTestCase):

    target_method = 'tomaxint'

    def test_tomaxint_none(self):
        x = self.generate()
        self.assertEqual(x.shape, ())
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())

    def test_tomaxint_int(self):
        x = self.generate(3)
        self.assertEqual(x.shape, (3,))
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())

    def test_tomaxint_tuple(self):
        x = self.generate((2, 3))
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())


@testing.parameterize(
    {'a': 3, 'size': 2, 'p': None},
    {'a': 3, 'size': 2, 'p': [0.3, 0.3, 0.4]},
    {'a': 3, 'size': (5, 5), 'p': [0.3, 0.3, 0.4]},
    {'a': 3, 'size': (5, 5), 'p': numpy.array([0.3, 0.3, 0.4])},
    {'a': 3, 'size': (), 'p': None},
    {'a': numpy.array([0.0, 1.0, 2.0]), 'size': 2, 'p': [0.3, 0.3, 0.4]},
)
@testing.fix_random()
@testing.gpu
class TestChoice1(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, p=self.p)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int64'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        vals = self.generate_many(
            a=self.a, size=self.size, p=self.p, _count=20)
        vals = [val.get() for val in vals]
        size_ = self.size if isinstance(self.size, tuple) else (self.size,)
        for val in vals:
            self.assertEqual(val.shape, size_)
        self.assertEqual(min(val.min() for val in vals), 0)
        self.assertEqual(max(val.max() for val in vals), 2)


@testing.parameterize(
    {'a': [0, 1, 2], 'size': 2, 'p': [0.3, 0.3, 0.4]},
)
@testing.fix_random()
@testing.gpu
class TestChoice2(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, p=self.p)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        vals = self.generate_many(
            a=self.a, size=self.size, p=self.p, _count=20)
        vals = [val.get() for val in vals]
        size_ = self.size if isinstance(self.size, tuple) else (self.size,)
        for val in vals:
            self.assertEqual(val.shape, size_)
        self.assertEqual(min(val.min() for val in vals), 0)
        self.assertEqual(max(val.max() for val in vals), 2)


@testing.fix_random()
@testing.gpu
class TestChoiceChi(RandomGeneratorTestCase):

    target_method = 'choice'

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        trial = 100
        vals = self.generate_many(3, 1, True, [0.3, 0.3, 0.4], _count=trial)
        vals = [val.get() for val in vals]
        counts = numpy.histogram(vals, bins=numpy.arange(4))[0]
        expected = numpy.array([30, 30, 40])
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3, 10)
    def test_goodness_of_fit_2(self):
        vals = self.generate(3, (5, 20), True, [0.3, 0.3, 0.4]).get()
        counts = numpy.histogram(vals, bins=numpy.arange(4))[0]
        expected = numpy.array([30, 30, 40])
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.fix_random()
@testing.gpu
class TestChoiceMultinomial(unittest.TestCase):

    @condition.repeat(3, 10)
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=0.02)
    def test_choice_multinomial(self, xp, dtype):
        p = xp.array([0.5, 0.25, 0.125, 0.125], dtype)
        trial = 10000
        x = xp.random.choice(len(p), trial, p=p)
        y = xp.bincount(x).astype('f') / trial
        return y


@testing.parameterize(
    {'a': 3.1, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': None, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': -3, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': [[0, 1], [2]], 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': [], 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': 3, 'size': 1, 'p': [[0.1, 0.1], [0.8]]},
    {'a': 2, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': 3, 'size': 1, 'p': [-0.1, 0.3, 0.8]},
    {'a': 3, 'size': 1, 'p': [0.1, 0.1, 0.7]},
)
@testing.fix_random()
@testing.gpu
class TestChoiceFailure(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def test_choice_invalid_value(self):
        with self.assertRaises(ValueError):
            self.rs.choice(a=self.a, size=self.size, p=self.p)


@testing.parameterize(
    {'a': 5, 'size': 2},
    {'a': 5, 'size': (2, 2)},
    {'a': 5, 'size': ()},
    {'a': numpy.array([0.0, 2.0, 4.0]), 'size': 2},
)
@testing.fix_random()
@testing.gpu
class TestChoiceReplaceFalse(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, replace=False)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        val = self.generate(a=self.a, size=self.size, replace=False).get()
        size = self.size if isinstance(self.size, tuple) else (self.size,)
        self.assertEqual(val.shape, size)
        self.assertTrue((0 <= val).all())
        self.assertTrue((val < 5).all())
        val = numpy.asarray(val)
        self.assertEqual(numpy.unique(val).size, val.size)


@testing.gpu
@testing.fix_random()
class TestGumbel(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'gumbel'

    def test_gumbel_1(self):
        self.generate()

    def test_gumbel_2(self):
        self.generate(0.0, 1.0, size=(3, 2))


@testing.gpu
@testing.fix_random()
class TestRandint(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'randint'

    def test_randint_1(self):
        self.generate(3)

    def test_randint_2(self):
        self.generate(3, 4, size=(3, 2))


@testing.gpu
@testing.fix_random()
class TestUniform(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'uniform'

    def test_uniform_1(self):
        self.generate()

    def test_uniform_2(self):
        self.generate(-4.2, 2.4, size=(3, 2))


@testing.parameterize(
    {'a': 3, 'size': 5},
    {'a': [1, 2, 3], 'size': 5},
)
@testing.fix_random()
@testing.gpu
class TestChoiceReplaceFalseFailure(unittest.TestCase):

    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_choice_invalid_value(self, xp):
        rs = xp.random.RandomState(seed=testing.generate_seed())
        rs.choice(a=self.a, size=self.size, replace=False)


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
class TestSetRandomState(unittest.TestCase):

    def setUp(self):
        self.rs_tmp = generator._random_states

    def tearDown(self, *args):
        generator._random_states = self.rs_tmp

    def test_set_random_state(self):
        rs = generator.RandomState()
        generator.set_random_state(rs)
        assert generator.get_random_state() is rs

    def test_set_random_state_call_multiple_times(self):
        generator.set_random_state(generator.RandomState())
        rs = generator.RandomState()
        generator.set_random_state(rs)
        assert generator.get_random_state() is rs


@testing.gpu
class TestRandomStateThreadSafe(unittest.TestCase):

    def setUp(self):
        cupy.random.reset_states()

    def test_get_random_state_thread_safe(self):
        seed = 10
        threads = [
            threading.Thread(target=lambda: cupy.random.seed(seed)),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = cupy.random.uniform()
        cupy.random.seed(seed)
        expected = cupy.random.uniform()
        self.assertEqual(actual, expected)

    def test_set_random_state_thread_safe(self):
        rs = cupy.random.RandomState()
        threads = [
            threading.Thread(target=lambda: cupy.random.set_random_state(rs)),
            threading.Thread(target=lambda: cupy.random.set_random_state(rs)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cupy.random.get_random_state() is rs


@testing.gpu
class TestGetRandomState2(unittest.TestCase):

    def setUp(self):
        self.rs_dict = generator._random_states
        generator._random_states = {}
        self.cupy_seed = os.getenv('CUPY_SEED')
        self.chainer_seed = os.getenv('CHAINER_SEED')

    def tearDown(self, *args):
        generator._random_states = self.rs_dict
        if self.cupy_seed is None:
            os.environ.pop('CUPY_SEED', None)
        else:
            os.environ['CUPY_SEED'] = self.cupy_seed
        if self.chainer_seed is None:
            os.environ.pop('CHAINER_SEED', None)
        else:
            os.environ['CHAINER_SEED'] = self.chainer_seed

    def test_get_random_state_no_cupy_no_chainer_seed(self):
        os.environ.pop('CUPY_SEED', None)
        os.environ.pop('CHAINER_SEED', None)
        rvs0 = self._get_rvs_reset()
        rvs1 = self._get_rvs_reset()

        self._check_different(rvs0, rvs1)

    def test_get_random_state_no_cupy_with_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(5))

        os.environ.pop('CUPY_SEED', None)
        os.environ['CHAINER_SEED'] = '5'
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def test_get_random_state_with_cupy_no_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(6))

        os.environ['CUPY_SEED'] = '6'
        os.environ.pop('CHAINER_SEED', None)
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def test_get_random_state_with_cupy_with_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(7))

        os.environ['CUPY_SEED'] = '7'
        os.environ['CHAINER_SEED'] = '8'
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def _get_rvs(self, rs):
        rvu = rs.rand(4)
        rvn = rs.randn(4)
        return rvu, rvn

    def _get_rvs_reset(self):
        generator.reset_states()
        return self._get_rvs(generator.get_random_state())

    def _check_same(self, rvs0, rvs1):
        for rv0, rv1 in zip(rvs0, rvs1):
            testing.assert_array_equal(rv0, rv1)

    def _check_different(self, rvs0, rvs1):
        for rv0, rv1 in zip(rvs0, rvs1):
            for r0, r1 in zip(rv0, rv1):
                self.assertNotEqual(r0, r1)


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
