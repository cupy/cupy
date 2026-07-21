from __future__ import annotations

from itertools import combinations
import sys

import numpy
import pytest

import cupy
from cupy import _environment
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cub_reduction
from cupy.cuda import memory


def expected_cub_block_kernel_calls(shape, axis):
    """Return how many generic CUB block-reduction kernels should launch."""
    if len(axis) == len(shape):
        return 2  # two-pass full reduction
    contiguous_size = int(numpy.prod([shape[i] for i in axis]))
    group_threads = _cub_reduction._get_cub_segment_group_threads(
        contiguous_size, 512)
    return 0 if group_threads == 0 else 1

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


@pytest.mark.parametrize(
    "shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
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

        a = cupy.random.random((10, 128))
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
                func_name, wraps=func, times_called=0):  # fallback
            a[:, :1].sum(axis=1)
        with testing.AssertFunctionIsCalled(
                func_name, wraps=func, times_called=0):  # not used
            a.sum(axis=0)

        _accelerator.set_routine_accelerators(old_routine_accelerators)


class TestSimpleCubReductionKernelSegmentMode(CubReductionTestBase):

    @pytest.mark.parametrize('contiguous_size,expected', [
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (64, 0),
        (128, 16),
        (256, 16),
        (512, 16),
        (2048, 16),
        (4096, 512),
    ])
    def test_get_cub_segment_group_threads(
            self, contiguous_size, expected):
        result = _cub_reduction._get_cub_segment_group_threads(
            contiguous_size, 512)
        assert result == expected

    @pytest.mark.parametrize('override,expected', [
        ('fallback', 0),
        ('block', 512),
        ('16', 16),
    ])
    def test_get_cub_segment_group_threads_override(
            self, monkeypatch, override, expected):
        monkeypatch.setenv('CUPY_CUB_REDUCTION_GROUP_THREADS', override)
        result = _cub_reduction._get_cub_segment_group_threads(64, 512)
        assert result == expected


class TestSimpleCubReductionKernelCorrectness(CubReductionTestBase):

    @pytest.fixture(autouse=True)
    def disable_routine_accelerators(self):
        self.old_routine_accelerators = (
            _accelerator.get_routine_accelerators())
        _accelerator.set_routine_accelerators([])
        yield
        _accelerator.set_routine_accelerators(
            self.old_routine_accelerators)

    @pytest.mark.parametrize('shape', [
        (32, 2),
        (17, 33),
        (9, 129),
    ])
    @pytest.mark.parametrize('op', [
        'nansum',
        'nanprod',
        'all',
        'any',
        'argmin',
        'argmax',
    ])
    def test_generic_cub_short_axis_correctness(self, shape, op):
        if op in ('all', 'any'):
            a = cupy.arange(numpy.prod(shape)).reshape(shape) % 3 == 0
            expect = getattr(numpy, op)(cupy.asnumpy(a), axis=1)
        else:
            a = testing.shaped_arange(shape, cupy, dtype=cupy.float32)
            if op in ('nansum', 'nanprod'):
                a[:, 0] = cupy.nan
            expect = getattr(numpy, op)(cupy.asnumpy(a), axis=1)

        actual = getattr(cupy, op)(a, axis=1)
        testing.assert_allclose(actual, expect)

    @pytest.mark.parametrize('shape', [
        (32, 2),
        (17, 33),
        (9, 129),
    ])
    def test_generic_user_reduction_kernel_correctness(self, shape):
        kernel = cupy.ReductionKernel(
            'float32 x', 'float32 y',
            'x * x', 'a + b', 'y = a', '0',
            'cupy_test_square_sum')
        a = testing.shaped_arange(shape, cupy, dtype=cupy.float32)

        actual = kernel(a, axis=1)
        expect = (cupy.asnumpy(a) ** 2).sum(axis=1)
        testing.assert_allclose(actual, expect)
