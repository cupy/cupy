import unittest
import warnings

import numpy

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr


class TestDummyDeviceType(unittest.TestCase):

    def test_int(self):
        self.assertEqual(int(cuda.DummyDeviceType()), -1)

    def test_eq(self):
        self.assertEqual(cuda.DummyDeviceType(), cuda.DummyDeviceType())

    def test_ne(self):
        self.assertNotEqual(cuda.DummyDeviceType(), 1)


_builtins_available = False
try:
    import builtins
    _builtins_available = True
except ImportError:
    pass


class TestCuda(unittest.TestCase):

    def test_get_dummy_device(self):
        self.assertIs(cuda.get_device(), cuda.DummyDevice)

    def test_get_device_for_numpy_int(self):
        self.assertIs(cuda.get_device(numpy.int64(0)), cuda.DummyDevice)

    @attr.gpu
    def test_get_dummy_device_for_empty_array(self):
        x = cuda.cupy.array([]).reshape((0, 10))
        self.assertIs(cuda.get_device(x), cuda.DummyDevice)

    @attr.gpu
    def test_get_device_for_int(self):
        self.assertEqual(cuda.get_device(0), cuda.Device(0))

    @attr.gpu
    @unittest.skipUnless(_builtins_available,
                         'builtins module is not available')
    def test_get_device_for_builtin_int(self):
        # builtins.int is from future package and it is different
        # from builtin int/long on Python 2.
        self.assertEqual(cuda.get_device(builtins.int(0)), cuda.Device(0))

    @attr.gpu
    def test_get_device_for_device(self):
        device = cuda.get_device(0)
        self.assertIs(cuda.get_device(device), device)

    def test_to_gpu_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                cuda.to_gpu(x)

    def test_get_array_module_for_numpy(self):
        self.assertIs(cuda.get_array_module(numpy.array([])), numpy)
        self.assertIs(
            cuda.get_array_module(chainer.Variable(numpy.array([]))),
            numpy)

    @attr.gpu
    def test_get_array_module_for_cupy(self):
        self.assertIs(cuda.get_array_module(cuda.cupy.array([])), cuda.cupy)
        self.assertIs(
            cuda.get_array_module(chainer.Variable(cuda.cupy.array([]))),
            cuda.cupy)

    def test_empy_unavailable(self):
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                with warnings.catch_warnings():
                    cuda.empty(())

    def test_empy_like_unavailable(self):
        x = numpy.array([1])
        if not cuda.available:
            with self.assertRaises(RuntimeError):
                with warnings.catch_warnings():
                    cuda.empty_like(x)


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToCPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))

    def test_numpy_array(self):
        y = cuda.to_cpu(self.x)
        self.assertIs(self.x, y)  # Do not copy

    @attr.gpu
    def test_cupy_array(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x)
        self.assertIsInstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        with cuda.Device(0):
            x = cuda.to_gpu(self.x)
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with cuda.Device(1):
            y = cuda.to_cpu(x)
        self.assertIsInstance(y, numpy.ndarray)
        numpy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_numpy_array_async(self):
        y = cuda.to_cpu(self.x, stream=cuda.Stream())
        self.assertIsInstance(y, numpy.ndarray)
        self.assertIs(self.x, y)  # Do not copy

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=1)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_cpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, numpy.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    def test_variable(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(TypeError):
            cuda.to_cpu(x)


class TestWorkspace(unittest.TestCase):

    def setUp(self):
        self.space = cuda.get_max_workspace_size()

    def tearDown(self):
        cuda.set_max_workspace_size(self.space)

    def test_size(self):
        size = 1024
        cuda.set_max_workspace_size(size)
        self.assertEqual(size, cuda.get_max_workspace_size())


@testing.parameterize(
    {'c_contiguous': True},
    {'c_contiguous': False},
)
class TestToGPU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3))
        if not self.c_contiguous:
            self.x = self.x.T

    @attr.gpu
    def test_numpy_array(self):
        y = cuda.to_gpu(self.x)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.gpu
    def test_cupy_array1(self):
        x = cuda.to_gpu(self.x)
        y = cuda.to_gpu(x)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIs(x, y)  # Do not copy

    @attr.multi_gpu(2)
    def test_cupy_array2(self):
        x = cuda.to_gpu(self.x, device=0)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_gpu(x, device=1)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertEqual(int(y.device), 1)

    @attr.gpu
    def test_numpy_array_async(self):
        y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)

    @attr.multi_gpu(2)
    def test_numpy_array_async2(self):
        y = cuda.to_gpu(self.x, device=1, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        self.assertEqual(int(y.device), 1)

    @attr.multi_gpu(2)
    def test_numpy_array_async3(self):
        with cuda.Device(1):
            y = cuda.to_gpu(self.x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        cuda.cupy.testing.assert_array_equal(self.x, y)
        self.assertEqual(int(y.device), 1)

    @attr.gpu
    def test_cupy_array_async1(self):
        x = cuda.to_gpu(self.x)
        if not self.c_contiguous:
            x = cuda.cupy.asfortranarray(x)
        y = cuda.to_gpu(x, stream=cuda.Stream())
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIs(x, y)  # Do not copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async2(self):
        x = cuda.to_gpu(self.x, device=0)
        with x.device:
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        y = cuda.to_gpu(x, device=1, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIsNot(x, y)  # Do copy
        cuda.cupy.testing.assert_array_equal(x, y)

    @attr.multi_gpu(2)
    def test_cupy_array_async3(self):
        with cuda.Device(0):
            x = cuda.to_gpu(self.x)
            if not self.c_contiguous:
                x = cuda.cupy.asfortranarray(x)
        with cuda.Device(1):
            y = cuda.to_gpu(x, stream=cuda.Stream.null)
        self.assertIsInstance(y, cuda.ndarray)
        self.assertIsNot(x, y)  # Do copy
        cuda.cupy.testing.assert_array_equal(x, y)

    def test_variable_cpu(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(TypeError):
            cuda.to_cpu(x)


testing.run_module(__name__, __file__)


class TestFusion(unittest.TestCase):

    @cuda.fuse()
    def sample_function(x, y, z):
        return cuda.square(cuda.add(x, y))

    def random_bool(self):
        return numpy.random.randint(0, 1, (10, 10)) == 0

    def random_int(self, lower=-1000, higher=1000):
        return numpy.random.randint(lower, higher, (10, 10))

    def random_real(self, lower=-1000, higher=1000):
        return numpy.random.rand(10, 10) * (higher - lower) + lower

    def check(self, func, n, gen, *args):

        @cuda.fuse(input_num=n)
        def f(*x):
            return func(*x)

        if type(gen) == tuple:
            ndata = [g(*a) for i, g, a in zip(range(n), list(gen), args)]
        else:
            ndata = [gen(*args) for i in range(n)]
        nret = func(*ndata)
        fnret = f(*ndata)
        nret = list(nret) if type(nret) == tuple else [nret]
        fnret = list(fnret) if type(fnret) == tuple else [fnret]
        for n, fn in zip(nret, fnret):
            numpy.testing.assert_array_almost_equal(n, fn)

        if cuda.available:
            cdata = map(cuda.to_gpu, ndata)
            cret = func(*cdata)
            fcret = f(*cdata)
            cret = list(cret) if type(cret) == tuple else [cret]
            fcret = list(fcret) if type(fcret) == tuple else [fcret]
            for n, c, fc in zip(nret, cret, fcret):
                numpy.testing.assert_array_almost_equal(n, cuda.to_cpu(c))
                numpy.testing.assert_array_almost_equal(n, cuda.to_cpu(fc))

    def check_reduce(self, func, n, reduce_f, gen, *args):

        @cuda.fuse(input_num=n, reduce=reduce_f)
        def f(*x):
            return func(*x)

        ndata = [gen(*args) for i in range(n)]
        fnret = f(*ndata)
        if cuda.available:
            cdata = map(cuda.to_gpu, ndata)
            fcret = f(*cdata)
            numpy.testing.assert_array_almost_equal(fnret, cuda.to_cpu(fcret))

    def test_bitwise(self):
        self.check(cuda.bitwise_and, 2, self.random_int)
        self.check(cuda.bitwise_or, 2, self.random_int)
        self.check(cuda.bitwise_xor, 2, self.random_int)
        self.check(cuda.invert, 1, self.random_int)
        self.check(cuda.left_shift, 2, self.random_int, 0, 20)
        self.check(cuda.right_shift, 2, self.random_int, 0, 20)

    def test_compare(self):
        self.check(cuda.greater, 2, self.random_int)
        self.check(cuda.greater_equal, 2, self.random_int)
        self.check(cuda.less, 2, self.random_int)
        self.check(cuda.less_equal, 2, self.random_int)
        self.check(cuda.equal, 2, self.random_int)
        self.check(cuda.not_equal, 2, self.random_int)

    def test_logic_content(self):
        self.check(cuda.isfinite, 1, self.random_real)
        self.check(cuda.isinf, 1, self.random_real)
        self.check(cuda.isnan, 1, self.random_real)

    def test_logic_ops(self):
        self.check(cuda.logical_and, 2, self.random_int, 0, 2)
        self.check(cuda.logical_or, 2, self.random_int, 0, 2)
        self.check(cuda.logical_not, 1, self.random_int, 0, 2)
        self.check(cuda.logical_xor, 2, self.random_int, 0, 2)

    def test_trigonometric(self):
        self.check(cuda.sin, 1, self.random_real)
        self.check(cuda.cos, 1, self.random_real)
        self.check(cuda.tan, 1, self.random_real)
        self.check(cuda.arcsin, 1, self.random_real, -1, 1)
        self.check(cuda.arccos, 1, self.random_real, -1, 1)
        self.check(cuda.arctan, 1, self.random_real)
        self.check(cuda.hypot, 2, self.random_real)
        self.check(cuda.deg2rad, 1, self.random_real)
        self.check(cuda.rad2deg, 1, self.random_real)
        self.check(cuda.degrees, 1, self.random_real)
        self.check(cuda.radians, 1, self.random_real)

    def test_hyperbolic(self):
        self.check(cuda.sinh, 1, self.random_real, -10, 10)
        self.check(cuda.cosh, 1, self.random_real, -10, 10)
        self.check(cuda.tanh, 1, self.random_real, -10, 10)
        self.check(cuda.arcsinh, 1, self.random_real, -10, 10)
        self.check(cuda.arccosh, 1, self.random_real, 1, 10)
        self.check(cuda.arctanh, 1, self.random_real, 0, 1)

    def test_rounding(self):
        self.check(cuda.rint, 1, self.random_real)
        self.check(cuda.floor, 1, self.random_real)
        self.check(cuda.ceil, 1, self.random_real)
        self.check(cuda.trunc, 1, self.random_real)

    def test_explog(self):
        self.check(cuda.exp, 1, self.random_real, -10, 10)
        self.check(cuda.expm1, 1, self.random_real, -10, 10)
        self.check(cuda.exp2, 1, self.random_real, -10, 10)
        self.check(cuda.log, 1, self.random_real, 0, 10)
        self.check(cuda.log10, 1, self.random_real, 0, 10)
        self.check(cuda.log2, 1, self.random_real, 0, 10)
        self.check(cuda.log1p, 1, self.random_real, -1, 10)
        self.check(cuda.logaddexp, 2, self.random_real, 0, 10)
        self.check(cuda.logaddexp2, 2, self.random_real, 0, 10)

    def test_floating(self):
        self.check(cuda.signbit, 1, self.random_real)
        self.check(cuda.copysign, 2, self.random_real)
        self.check(cuda.ldexp, 2, self.random_int, 1, 10)
        self.check(cuda.frexp, 1, self.random_real, 1, 1000)
        self.check(cuda.nextafter, 2, self.random_real)

    def test_arithmetic(self):
        self.check(cuda.add, 2, self.random_real)
        self.check(cuda.reciprocal, 1, self.random_real)
        self.check(cuda.negative, 1, self.random_real)
        self.check(cuda.multiply, 2, self.random_real)
        self.check(cuda.divide, 2, self.random_real)
        self.check(cuda.power, 2, self.random_real, 0, 10)
        self.check(cuda.subtract, 2, self.random_real)
        self.check(cuda.true_divide, 2, self.random_int, 1, 1000)
        self.check(cuda.floor_divide, 2, self.random_real, 1, 1000)
        self.check(cuda.fmod, 2, self.random_real)
        self.check(cuda.mod, 2, self.random_int, 1, 1000)
        self.check(cuda.modf, 1, self.random_real)
        self.check(cuda.remainder, 2, self.random_int, 1, 1000)

    def test_misc(self):
        self.check(cuda.sqrt, 1, self.random_real, 0, 1000)
        self.check(cuda.sqrt_fixed, 1, self.random_real, 0, 1000)
        self.check(cuda.square, 1, self.random_real)
        self.check(cuda.absolute, 1, self.random_real)
        self.check(cuda.abs, 1, self.random_real)
        self.check(cuda.sign, 1, self.random_real)
        self.check(cuda.maximum, 2, self.random_real)
        self.check(cuda.minimum, 2, self.random_real)
        self.check(cuda.fmax, 2, self.random_real)
        self.check(cuda.fmin, 2, self.random_real)

    def test_special(self):
        self.check(cuda.where, 3,
                   (self.random_bool, self.random_int, self.random_int),
                   (), (0, 100), (0, 100))
        self.check(cuda.clip, 3,
                   (self.random_real, self.random_real, self.random_real),
                   (0, 1000), (0, 500), (500, 1000))

    def test_reduce(self):
        self.check_reduce(cuda.bitwise_and, 2, cuda.sum, self.random_int)
        self.check_reduce(cuda.sqrt, 1, cuda.prod, self.random_int, 1, 2)
        self.check_reduce(cuda.sqrt, 1, cuda.prod, self.random_real, 1, 2)
        self.check_reduce(lambda x: x, 1, cuda.amax, self.random_int)
        self.check_reduce(lambda x: x, 1, cuda.amin, self.random_int)
