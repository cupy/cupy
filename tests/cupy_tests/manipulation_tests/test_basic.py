import itertools
import warnings

import numpy
import pytest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestBasic:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_different_contiguity(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 2), xp, dtype).T
        b = xp.empty((2, 3, 2), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype='?')
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_broadcast(self, xp, dtype):
        a = testing.shaped_arange((3, 1), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @pytest.mark.parametrize(('dst_shape', 'src_shape'), [
        ((), (2,)),
        ((2, 0, 5, 4), (2, 0, 3, 4)),
        ((6,), (2, 3)),
        ((2, 3), (6,)),
    ])
    def test_copyto_raises_shape(self, dst_shape, src_shape):
        for xp in (numpy, cupy):
            dst = xp.zeros(dst_shape, int)
            src = xp.zeros(src_shape, int)
            with pytest.raises(ValueError):
                xp.copyto(dst, src)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze(self, xp, dtype):
        a = testing.shaped_arange((1, 1, 3, 4), xp, dtype)
        b = xp.empty((3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze_different_contiguity(self, xp, dtype):
        a = testing.shaped_arange((1, 1, 3, 4), xp, dtype)
        b = xp.empty((4, 3), dtype=dtype).T
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_squeeze_broadcast(self, xp, dtype):
        a = testing.shaped_arange((1, 2, 1, 4), xp, dtype)
        b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_where(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xp, dtype)
        c = testing.shaped_arange((2, 3, 4), xp, '?')
        xp.copyto(a, b, where=c)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_where_squeeze_broadcast(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((1, 2, 1, 4), xp, dtype)
        c = testing.shaped_arange((3, 4), xp, '?')
        xp.copyto(a, b, where=c)
        return a

    @pytest.mark.parametrize('shape', [(2, 3, 4), (0,)])
    @testing.for_all_dtypes(no_bool=True)
    def test_copyto_where_raises(self, dtype, shape):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(shape, xp, 'i')
            b = testing.shaped_reverse_arange(shape, xp, 'i')
            c = testing.shaped_arange(shape, xp, dtype)
            with pytest.raises(TypeError):
                xp.copyto(a, b, where=c)

    def _check_copyto_where_multigpu_raises(self, dtype, ngpus):
        def get_numpy():
            a = testing.shaped_arange((2, 3, 4), numpy, dtype)
            b = testing.shaped_reverse_arange((2, 3, 4), numpy, dtype)
            c = testing.shaped_arange((2, 3, 4), numpy, '?')
            numpy.copyto(a, b, where=c)
            return a

        for dev1, dev2, dev3, dev4 in itertools.product(*[range(ngpus)] * 4):
            if dev1 == dev2 == dev3 == dev4:
                continue
            if not dev1 <= dev2 <= dev3 <= dev4:
                continue

            with cuda.Device(dev1):
                a = testing.shaped_arange((2, 3, 4), cupy, dtype)
            with cuda.Device(dev2):
                b = testing.shaped_reverse_arange((2, 3, 4), cupy, dtype)
            with cuda.Device(dev3):
                c = testing.shaped_arange((2, 3, 4), cupy, '?')
            with cuda.Device(dev4):
                if all([(peer == dev4) or
                        (cuda.runtime.deviceCanAccessPeer(dev4, peer) == 1)
                        for peer in (dev1, dev2, dev3)]):
                    with pytest.warns(cupy._util.PerformanceWarning):
                        cupy.copyto(a, b, where=c)
                else:
                    with pytest.raises(
                            ValueError,
                            match='Peer access is unavailable'):
                        cupy.copyto(a, b, where=c)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copyto_where_multigpu_raises(self, dtype):
        self._check_copyto_where_multigpu_raises(dtype, 2)

    @testing.multi_gpu(4)
    @testing.for_all_dtypes()
    def test_copyto_where_multigpu_raises_4(self, dtype):
        self._check_copyto_where_multigpu_raises(dtype, 4)

    @testing.multi_gpu(6)
    @testing.for_all_dtypes()
    def test_copyto_where_multigpu_raises_6(self, dtype):
        self._check_copyto_where_multigpu_raises(dtype, 6)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_multigpu(self, xp, dtype):
        with cuda.Device(0):
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
        with cuda.Device(1):
            b = xp.empty((2, 3, 4), dtype=dtype)
        xp.copyto(b, a)
        return b

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copyto_multigpu_noncontinguous(self, dtype):
        with cuda.Device(0):
            src = testing.shaped_arange((2, 3, 4), cupy, dtype)
            src = src.swapaxes(0, 1)
        with cuda.Device(1):
            dst = cupy.empty_like(src)
            cupy.copyto(dst, src)

        expected = testing.shaped_arange((2, 3, 4), numpy, dtype)
        expected = expected.swapaxes(0, 1)

        testing.assert_array_equal(expected, src.get())
        testing.assert_array_equal(expected, dst.get())


@testing.parameterize(
    *testing.product(
        {'src': [float(3.2), int(0), int(4), int(-4), True, False, 1 + 1j],
         'dst_shape': [(), (0,), (1,), (1, 1), (2, 2)]}))
@testing.gpu
class TestCopytoFromScalar:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        xp.copyto(dst, self.src)
        return dst

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto_where(self, xp, dtype):
        dst = xp.ones(self.dst_shape, dtype=dtype)
        mask = (testing.shaped_arange(
            self.dst_shape, xp, dtype) % 2).astype(xp.bool_)
        xp.copyto(dst, self.src, where=mask)
        return dst


@pytest.mark.parametrize(
    'casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
class TestCopytoFromNumpyScalar:

    @testing.for_all_dtypes_combination(('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto(self, xp, dtype1, dtype2, casting):
        dst = xp.zeros((2, 3, 4), dtype=dtype1)
        src = numpy.array(1, dtype=dtype2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            xp.copyto(dst, src, casting)
        return dst

    @testing.for_all_dtypes()
    @pytest.mark.parametrize('make_src',
                             [lambda dtype: numpy.array([1], dtype=dtype),
                              lambda dtype: dtype(1)])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto2(self, xp, make_src, dtype, casting):
        dst = xp.zeros((2, 3, 4), dtype=dtype)
        src = make_src(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            xp.copyto(dst, src, casting)
        return dst

    @testing.for_all_dtypes_combination(('dtype1', 'dtype2'))
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def test_copyto_where(self, xp, dtype1, dtype2, casting):
        shape = (2, 3, 4)
        dst = xp.ones(shape, dtype=dtype1)
        src = numpy.array(1, dtype=dtype2)
        mask = (testing.shaped_arange(shape, xp, dtype1) % 2).astype(xp.bool_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            xp.copyto(dst, src, casting=casting, where=mask)
        return dst


@pytest.mark.parametrize('shape', [(3, 2), (0,)])
@pytest.mark.parametrize('where', [
    float(3.2), int(0), int(4), int(-4), True, False, 1 + 1j
])
@testing.for_all_dtypes()
@testing.numpy_cupy_allclose()
def test_copyto_scalarwhere(xp, dtype, where, shape):
    dst = xp.zeros(shape, dtype=dtype)
    src = xp.ones(shape, dtype=dtype)
    xp.copyto(dst, src, where=where)
    return dst
