from __future__ import annotations

import concurrent.futures

import numpy
import pytest

import cupy
from cupy import testing
try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None


class C(cupy.ndarray):

    def __new__(cls, *args, info=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


class TestArrayUfunc:

    @testing.for_all_dtypes()
    def test_unary_op(self, dtype):
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.sin(a)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b = a.get()
        outb = numpy.sin(b)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_unary_op_out(self, dtype):
        a = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        b = a.get()
        outb = numpy.sin(b)
        # pre-make output with same type as input
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=outb.dtype)
        numpy.sin(a, out=outa)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = numpy.add(a1, a2)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op_out(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        outa = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        numpy.add(a1, a2, out=outa)
        b1 = a1.get()
        b2 = a2.get()
        outb = numpy.add(b1, b2)
        assert numpy.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_mixed_op(self, dtype):
        a1 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(numpy.array([0, 1, 2]), dtype=dtype).get()
        with pytest.raises(TypeError):
            # attempt to add cupy and numpy arrays
            numpy.add(a1, a2)
        with pytest.raises(TypeError):
            # check reverse order
            numpy.add(a2, a1)
        with pytest.raises(TypeError):
            # reject numpy output from cupy
            numpy.add(a1, a1, out=a2)
        with pytest.raises(TypeError):
            # reject cupy output from numpy
            numpy.add(a2, a2, out=a1)
        with pytest.raises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            numpy.sin(a1, out=())
        with pytest.raises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            numpy.sin(a1, out=(a1, a1))

    @testing.numpy_cupy_array_equal()
    def test_indexing(self, xp):
        a = cupy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = cupy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

    @testing.numpy_cupy_array_equal()
    def test_shares_memory(self, xp):
        a = cupy.testing.shaped_arange((1000, 1000), xp, 'int64')
        b = xp.transpose(a)
        a += b
        return a

    @pytest.mark.parametrize('shape', [(3,), (2, 3)])
    def test_subclass_unary_op(self, shape):
        a = cupy.arange(numpy.prod(shape)).reshape(shape).view(C)
        a.info = 1
        outa = cupy.sin(a)
        assert isinstance(outa, C)
        assert outa.info is not None and outa.info == 1

        b = a.get()
        outb = numpy.sin(b)
        testing.assert_allclose(outa, outb)

    @pytest.mark.parametrize('rows', [0, 2])
    def test_subclass_broadcast_template(self, rows):
        class Array(cupy.ndarray):
            def __array_finalize__(self, parent):
                self.parent = parent

        a = cupy.ones((1, 3)).view(Array)
        result = cupy.add(a, cupy.ones((rows, 3)))
        assert isinstance(result, Array)
        assert result.parent is a
        testing.assert_array_equal(result, numpy.full((rows, 3), 2.0))

    def test_subclass_binary_op(self):
        a0 = cupy.array([0, 1, 2]).view(C)
        a0.info = 1
        a1 = cupy.array([3, 4, 5]).view(C)
        a1.info = 2
        outa = cupy.add(a0, a1)
        assert isinstance(outa, C)
        # a0 is used to initialize outa.info
        assert outa.info is not None and outa.info == 1

        b0 = a0.get()
        b1 = a1.get()
        outb = numpy.add(b0, b1)
        testing.assert_allclose(outa, outb)

    def test_subclass_binary_op_mixed(self):
        a0 = cupy.array([0, 1, 2])
        a1 = cupy.array([3, 4, 5]).view(C)
        a1.info = 1
        outa = cupy.add(a0, a1)
        assert isinstance(outa, C)
        # The first appearance of C's instance is used to initialize outa.info
        assert outa.info is not None and outa.info == 1

        b0 = a0.get()
        b1 = a1.get()
        outb = numpy.add(b0, b1)
        testing.assert_allclose(outa, outb)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_outer(self, xp):
        a = cupy.testing.shaped_arange((3, 4), xp)
        b = cupy.testing.shaped_arange((5, 6), xp)
        return numpy.add.outer(a, b)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_at(self, xp):
        a = cupy.testing.shaped_arange((10,), xp)
        b = cupy.testing.shaped_arange((5,), xp)
        indices = xp.array([0, 3, 6, 7, 9])
        numpy.add.at(a, indices, b)
        return a

    @testing.numpy_cupy_array_equal()
    def test_ufunc_at_scalar(self, xp):
        a = cupy.testing.shaped_arange((10,), xp)
        b = 7
        indices = xp.array([0, 3, 6, 7, 9])
        numpy.add.at(a, indices, b)
        return a

    @testing.numpy_cupy_array_equal()
    def test_ufunc_reduce(self, xp):
        a = cupy.testing.shaped_arange((10, 12), xp)
        return numpy.add.reduce(a, axis=-1)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_accumulate(self, xp):
        a = cupy.testing.shaped_arange((10, 12), xp)
        return numpy.add.accumulate(a, axis=-1)

    @testing.numpy_cupy_array_equal()
    def test_ufunc_reduceat(self, xp):
        a = cupy.testing.shaped_arange((10, 12), xp)
        indices = xp.array([0, 3, 6, 7, 9])
        return numpy.add.reduceat(a, indices, axis=-1)


class TestUfunc:
    @pytest.mark.parametrize('ufunc', [
        'add',
        'sin',
    ])
    @testing.numpy_cupy_equal()
    def test_types(self, xp, ufunc):
        types = getattr(xp, ufunc).types
        if xp == numpy:
            assert isinstance(types, list)
            types = list(dict.fromkeys(  # remove dups: numpy/numpy#7897
                sig for sig in types
                # CuPy does not support the following dtypes:
                # (c)longdouble, datetime, timedelta, and object.
                if not any(t in sig for t in 'GgMmO')
            ))
        if xp == cupy and bfloat16 is not None:
            # Ugly, but fetch the char from the `loop` and remove it
            # (at the time of writing the char was E, hopefully it'll change)
            types = [t for t in types if numpy.dtype(bfloat16).char not in t]
        return types

    @testing.numpy_cupy_allclose()
    def test_unary_out_tuple(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        out = xp.zeros((2, 3), dtype)
        ret = xp.sin(a, out=(out,))
        assert ret is out
        return ret

    @testing.numpy_cupy_allclose()
    def test_unary_out_positional_none(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.sin(a, None)

    @testing.numpy_cupy_allclose()
    def test_binary_out_tuple(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        out = xp.zeros((2, 3), dtype)
        ret = xp.add(a, b, out=(out,))
        assert ret is out
        return ret

    @testing.numpy_cupy_allclose()
    def test_biary_out_positional_none(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        return xp.add(a, b, None)

    @testing.numpy_cupy_allclose()
    def test_divmod_out_tuple(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out0 = xp.zeros((2, 3), dtype)
        out1 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out=(out0, out1))
        assert ret[0] is out0
        assert ret[1] is out1
        return ret

    @testing.numpy_cupy_allclose()
    def test_divmod_out_positional_none(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = xp.ones((2, 3), dtype)
        return xp.divmod(a, b, None, None)

    @testing.numpy_cupy_allclose()
    def test_divmod_out_partial(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out0 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out0)  # out1 is None
        assert ret[0] is out0
        return ret

    @testing.numpy_cupy_allclose()
    def test_divmod_out_partial_tuple(self, xp):
        dtype = xp.float64
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        out1 = xp.zeros((2, 3), dtype)
        ret = xp.divmod(a, b, out=(None, out1))
        assert ret[1] is out1
        return ret

    @pytest.mark.parametrize('shape', [(), (6,), (2, 3), (1, 2, 1, 3), (0, 2)])
    @testing.for_dtypes([numpy.int32, numpy.float32, numpy.float64,
                         numpy.complex64])
    @testing.numpy_cupy_array_equal()
    def test_inplace_scalar(self, xp, dtype, shape):
        a = testing.shaped_arange(shape, xp, dtype)
        metadata = a.shape, a.strides, a.dtype
        ret = xp.add(a, 2, out=a)
        assert ret is a
        assert (a.shape, a.strides, a.dtype) == metadata
        return a

    @pytest.mark.parametrize('layout', [
        'contiguous', 'leading_singleton', 'broadcast', 'transpose',
        'reverse'])
    @testing.numpy_cupy_array_equal()
    def test_mixed_dtype_inputs(self, xp, layout):
        a = testing.shaped_arange((2, 3), xp, numpy.float32)
        if layout == 'contiguous':
            b = testing.shaped_arange((2, 3), xp, numpy.float64)
        elif layout == 'leading_singleton':
            b = testing.shaped_arange((1, 2, 3), xp, numpy.float64)
        elif layout == 'broadcast':
            b = xp.arange(3, dtype=xp.float64)
        elif layout == 'transpose':
            b = testing.shaped_arange((3, 2), xp, numpy.float64).T
        else:
            b = testing.shaped_arange((2, 3), xp, numpy.float64)[:, ::-1]
        return xp.add(a, b)

    @testing.numpy_cupy_array_equal()
    def test_overlapping_output(self, xp):
        a = testing.shaped_arange((3, 4), xp, numpy.int32)
        out = a[1:]
        ret = xp.add(a[:-1], 3, out=out)
        assert ret is out
        return a

    @testing.numpy_cupy_array_equal()
    def test_noncontiguous_output(self, xp):
        a = testing.shaped_arange((2, 3), xp, numpy.float32)
        out = xp.empty((3, 2), dtype=xp.float32).T
        ret = xp.add(a, 2, out=out)
        assert ret is out
        return out

    @testing.numpy_cupy_array_equal()
    def test_where(self, xp):
        a = testing.shaped_arange((2, 3), xp, numpy.float32)
        out = xp.full((2, 3), -1, dtype=xp.float32)
        where = xp.array([[True, False, True], [False, True, False]])
        kwargs = {'_where' if xp is cupy else 'where': where}
        return xp.add(a, 2, out=out, **kwargs)

    @pytest.mark.thread_unsafe(reason='modifies the core ndarray binding')
    @pytest.mark.parametrize('module_getattr', [False, True])
    def test_missing_ndarray_binding(self, monkeypatch, module_getattr):
        a = cupy.empty((2, 3))
        with monkeypatch.context() as patcher:
            patcher.delattr(cupy._core.core, 'ndarray')
            if module_getattr:
                def get_missing(name):
                    if name == 'ndarray':
                        return cupy.ndarray
                    raise AttributeError(name)

                patcher.setattr(cupy._core.core, '__getattr__', get_missing,
                                raising=False)
            with pytest.raises(
                    NameError, match="name 'ndarray' is not defined"):
                cupy.add(a, 1, out=a)

    @pytest.mark.thread_unsafe(reason='modifies the core module class')
    def test_module_attribute_hook(self, monkeypatch):
        core_module = cupy._core.core

        class CoreModule(type(core_module)):
            def __getattribute__(self, name):
                if name == 'ndarray':
                    raise AssertionError('unexpected module attribute lookup')
                return super().__getattribute__(name)

        a = cupy.zeros((2, 3), dtype=cupy.int32)
        with monkeypatch.context() as patcher:
            patcher.setattr(core_module, '__class__', CoreModule)
            ret = cupy.add(a, 1, out=a)
        assert ret is a
        testing.assert_array_equal(a, numpy.ones((2, 3), dtype=numpy.int32))

    @pytest.mark.thread_unsafe(reason='modifies ndarray bindings')
    def test_rebound_subclass_metaclass(self, monkeypatch):
        base = cupy.ndarray.__base__

        class Meta(type):
            def __getattribute__(cls, name):
                if name == '__base__':
                    return base
                return super().__getattribute__(name)

        class ReboundArray(cupy.ndarray, metaclass=Meta):
            def __new__(cls, *args, **kwargs):
                if kwargs.get('_no_init', False):
                    raise RuntimeError('reduced view constructor')
                return super().__new__(cls, *args, **kwargs)

        a = ReboundArray((2, 3), dtype=cupy.int32)
        with monkeypatch.context() as patcher:
            patcher.setattr(cupy._core.core, 'ndarray', ReboundArray)
            patcher.setattr(cupy, 'ndarray', ReboundArray)
            with pytest.raises(RuntimeError, match='reduced view constructor'):
                cupy.add(a, 1, out=a)

    @pytest.mark.thread_unsafe(reason='explicitly multithreaded test')
    def test_thread_local_temporary_inputs(self):
        device_id = cupy.cuda.runtime.getDevice()

        def worker(offset):
            with cupy.cuda.Device(device_id):
                with cupy.cuda.Stream(non_blocking=True):
                    a = cupy.arange(48, dtype=cupy.int32).reshape(6, 8)
                    for _ in range(20):
                        cupy.add(a, offset, out=a)
                    out = cupy.add(a, cupy.ones(a.shape, dtype=a.dtype))
                    del a
                    return out.get()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))
        for offset, result in enumerate(results):
            expected = numpy.arange(48).reshape(6, 8) + 20 * offset + 1
            numpy.testing.assert_array_equal(result, expected)

    @testing.slow
    def test_contiguous_large_index(self):
        try:
            a = cupy.zeros((2, 2**30 + 1), dtype=cupy.uint8)
        except MemoryError:
            pytest.skip('out of memory in test')
        cupy.add(a, 1, out=a)
        assert int(a.sum()) == a.size
