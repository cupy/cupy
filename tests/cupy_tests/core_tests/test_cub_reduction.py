from __future__ import annotations

from itertools import combinations
import sys

import pytest

import cupy
from cupy import _environment
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cub_reduction
from cupy.cuda import memory


# This test class and its children below only test if CUB backend can be used
# or not; they don't verify its correctness as it's already extensively covered
# by existing tests
class CubReductionTestBase:
    """
    Note: call self.can_use() when arrays are already allocated, otherwise
    call self._test_can_use().
    """

    @pytest.fixture(autouse=True)
    def configure(self):
        if _environment.get_cub_path() is None:
            pytest.skip('CUB not found')
        if cupy.cuda.runtime.is_hip:
            if _environment.get_hipcc_path() is None:
                pytest.skip('hipcc is not found')

        self.can_use = cupy._core._cub_reduction._can_use_cub_block_reduction

        self.old_accelerators = _accelerator.get_reduction_accelerators()
        _accelerator.set_reduction_accelerators(['cub'])
        yield
        _accelerator.set_reduction_accelerators(self.old_accelerators)

    def _test_can_use(
            self, i_shape, o_shape, r_axis, o_axis, order, expected):
        in_args = [cupy.testing.shaped_arange(i_shape, order=order), ]
        out_args = [cupy.testing.shaped_arange(o_shape, order=order), ]
        result = self.can_use(in_args, out_args, r_axis, o_axis) is not None
        assert result is expected


_MIN_SIZE = cupy._core._cub_reduction._CUB_REDUCE_SIZE_THRESHOLD


@pytest.mark.parametrize(
    "shape", [
        (_MIN_SIZE,), (_MIN_SIZE, _MIN_SIZE+1), (_MIN_SIZE, 3, _MIN_SIZE+1),
        (_MIN_SIZE, 3, 4, _MIN_SIZE+1)]
)
@pytest.mark.parametrize(
    "order", ['C', 'F'],
)
class TestSimpleCubReductionKernelContiguity(CubReductionTestBase):

    @testing.for_contiguous_axes()
    def test_can_use_cub_contiguous(self, axis, shape, order):
        r_axis = axis
        i_shape = shape
        o_axis = tuple(i for i in range(len(i_shape)) if i not in r_axis)
        o_shape = tuple(shape[i] for i in o_axis)
        self._test_can_use(i_shape, o_shape, r_axis, o_axis, order, True)

    @testing.for_contiguous_axes()
    def test_can_use_cub_non_contiguous(self, axis, shape, order):
        # array is contiguous, but reduce_axis is not
        dim = len(shape)
        r_dim = len(axis)
        non_contiguous_axes = [i for i in combinations(range(dim), r_dim)
                               if i != axis]

        i_shape = shape
        for r_axis in non_contiguous_axes:
            o_axis = tuple(i for i in range(dim) if i not in r_axis)
            o_shape = tuple(shape[i] for i in o_axis)
            self._test_can_use(i_shape, o_shape, r_axis, o_axis,
                               order, False)


class TestSimpleCubReductionKernelMisc(CubReductionTestBase):

    def test_can_use_cub_nonsense_input1(self):
        # two inputs are not allowed
        a = cupy.random.random((2, 3, 4))
        b = cupy.random.random((2, 3, 4))
        c = cupy.empty((2, 3, ))
        assert self.can_use([a, b], [c], (2,), (0, 1)) is None

    def test_can_use_cub_nonsense_input2(self):
        # reduce_axis and out_axis do not add up to full axis set
        self._test_can_use((2, 3, 4), (2, 3), (2,), (0,), 'C', False)

    def test_can_use_cub_nonsense_input3(self):
        # array is neither C- nor F- contiguous
        a = cupy.random.random((3, 4, 5))
        a = a[:, 0:-1:2, 0:-1:3]
        assert not a.flags.forc
        b = cupy.empty((3,))
        assert self.can_use([a], [b], (1, 2), (0,)) is None

    def test_can_use_cub_zero_size_input(self):
        self._test_can_use((2, 0, 3), (), (0, 1, 2), (), 'C', False)

    # We actually just wanna test shapes, no need to allocate large memory.
    def test_can_use_cub_oversize_input1(self):
        # full reduction with array size > 64 GB
        mem = memory.alloc(100)
        a = cupy.ndarray((2**6 * 1024**3 + 1,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is None

    def test_can_use_cub_oversize_input2(self):
        # full reduction with array size = 64 GB should work!
        mem = memory.alloc(100)
        a = cupy.ndarray((2**6 * 1024**3,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is not None

    def test_can_use_cub_oversize_input3(self):
        # full reduction with 2^63-1 elements
        mem = memory.alloc(100)
        max_num = sys.maxsize
        a = cupy.ndarray((max_num,), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (0,), ()) is None

    def test_can_use_cub_oversize_input4(self):
        # partial reduction with too many (2^31) blocks
        mem = memory.alloc(100)
        a = cupy.ndarray((2**31, 8), dtype=cupy.int8, memptr=mem)
        b = cupy.empty((), dtype=cupy.int8)
        assert self.can_use([a], [b], (1,), (0,)) is None

    @pytest.mark.thread_unsafe(
        reason="AssertFunctionIsCalled and accelerate mutation.")
    def test_can_use_accelerator_set_unset(self):
        # ensure we use CUB block reduction and not CUB device reduction
        old_routine_accelerators = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators([])

        a = cupy.random.random((10, _cub_reduction._CUB_REDUCE_SIZE_THRESHOLD))
        # this is the only function we can mock; the rest is cdef'd
        func_name = ''.join(('cupy._core._cub_reduction.',
                             '_SimpleCubReductionKernel_get_cached_function'))
        func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=2):  # two passes
            a.sum()
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=1):  # one pass
            a.sum(axis=1)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # not used
            a.sum(axis=0)

        _accelerator.set_routine_accelerators(old_routine_accelerators)


class TestCubReductionUserOutput(CubReductionTestBase):

    @pytest.fixture(autouse=True)
    def disable_routine_accelerators(self):
        old = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators([])
        yield
        _accelerator.set_routine_accelerators(old)

    @pytest.mark.parametrize('order', ['C', 'F'])
    @pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_user_output(self, order, layout, keepdims):
        import numpy

        shape = (128, 2, 3) if order == 'F' else (2, 3, 128)
        axis = 0 if order == 'F' else 2
        host = numpy.arange(768, dtype=numpy.float32).reshape(shape) % 17
        a = cupy.array(host, order=order)
        expected = numpy.nansum(host, axis=axis, keepdims=keepdims)
        if layout == 'strided':
            backing_shape = expected.shape[:-1] + (expected.shape[-1] * 2,)
            backing = cupy.full(backing_shape, -8192, dtype=cupy.float32)
            out = backing[..., ::2]
            expected_backing = numpy.full(backing_shape, -8192,
                                          dtype=numpy.float32)
            expected_backing[..., ::2] = expected
        else:
            backing = cupy.full(expected.shape, -8192, dtype=cupy.float32,
                                order=layout)
            out = backing
            expected_backing = expected
        metadata = out.shape, out.strides, out.data.ptr
        func_name = ('cupy._core._cub_reduction.'
                     '_SimpleCubReductionKernel_get_cached_function')
        func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=int(layout == order)):
            result = cupy.nansum(a, axis=axis, out=out, keepdims=keepdims)
        assert result is out
        assert (out.shape, out.strides, out.data.ptr) == metadata
        testing.assert_array_equal(out, expected)
        testing.assert_array_equal(backing, expected_backing)
        testing.assert_array_equal(a, host)

    def test_user_output_singleton_strides(self):
        a = cupy.ones((128, 2, 3), dtype=cupy.float32, order='F')
        backing = cupy.empty((2, 3), dtype=cupy.float32, order='F')
        out = cupy.ndarray((1, 2, 3), dtype=cupy.float32,
                           memptr=backing.data, strides=(128, 4, 8))
        assert out.flags.f_contiguous
        metadata = out.shape, out.strides, out.data.ptr
        result = cupy.nansum(a, axis=0, out=out, keepdims=True)
        assert result is out
        assert (out.shape, out.strides, out.data.ptr) == metadata
        testing.assert_array_equal(out, 128)

    @pytest.mark.parametrize('order', ['C', 'F'])
    def test_allocated_output_uses_cub(self, order):
        a = testing.shaped_arange((128, 2, 128), dtype=cupy.float32,
                                  order=order)
        axis = 0 if order == 'F' else 2
        func_name = ('cupy._core._cub_reduction.'
                     '_SimpleCubReductionKernel_get_cached_function')
        func = _cub_reduction._SimpleCubReductionKernel_get_cached_function
        with testing.AssertFunctionIsCalled(func_name, wraps=func):
            result = cupy.nansum(a, axis=axis)
        import numpy

        testing.assert_array_equal(result, numpy.nansum(cupy.asnumpy(a),
                                                        axis=axis))
