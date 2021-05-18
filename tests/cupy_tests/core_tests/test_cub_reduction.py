from itertools import combinations
import unittest
import sys

import cupy
from cupy import _environment
from cupy import testing
from cupy._core import _accelerator
from cupy._core import _cub_reduction
from cupy.cuda import memory


# This test class and its children below only test if CUB backend can be used
# or not; they don't verify its correctness as it's already extensively covered
# by existing tests
@unittest.skipIf(_environment.get_cub_path() is None, 'CUB not found')
class CubReductionTestBase(unittest.TestCase):
    """
    Note: call self.can_use() when arrays are already allocated, otherwise
    call self._test_can_use().
    """

    def setUp(self):
        if cupy.cuda.runtime.is_hip:
            if _environment.get_hipcc_path() is None:
                self.skipTest('hipcc is not found')

        self.can_use = cupy._core._cub_reduction._can_use_cub_block_reduction

        self.old_accelerators = _accelerator.get_reduction_accelerators()
        _accelerator.set_reduction_accelerators(['cub'])

    def tearDown(self):
        _accelerator.set_reduction_accelerators(self.old_accelerators)

    def _test_can_use(
            self, i_shape, o_shape, r_axis, o_axis, order, expected):
        in_args = [cupy.testing.shaped_arange(i_shape, order=order), ]
        out_args = [cupy.testing.shaped_arange(o_shape, order=order), ]
        result = self.can_use(in_args, out_args, r_axis, o_axis) is not None
        assert result is expected


@testing.parameterize(*testing.product({
    'shape': [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
    'order': ('C', 'F'),
}))
@testing.gpu
class TestSimpleCubReductionKernelContiguity(CubReductionTestBase):

    @testing.for_contiguous_axes()
    def test_can_use_cub_contiguous(self, axis):
        r_axis = axis
        i_shape = self.shape
        o_axis = tuple(i for i in range(len(i_shape)) if i not in r_axis)
        o_shape = tuple(self.shape[i] for i in o_axis)
        self._test_can_use(i_shape, o_shape, r_axis, o_axis, self.order, True)

    @testing.for_contiguous_axes()
    def test_can_use_cub_non_contiguous(self, axis):
        # array is contiguous, but reduce_axis is not
        dim = len(self.shape)
        r_dim = len(axis)
        non_contiguous_axes = [i for i in combinations(range(dim), r_dim)
                               if i != axis]

        i_shape = self.shape
        for r_axis in non_contiguous_axes:
            o_axis = tuple(i for i in range(dim) if i not in r_axis)
            o_shape = tuple(self.shape[i] for i in o_axis)
            self._test_can_use(i_shape, o_shape, r_axis, o_axis,
                               self.order, False)


@testing.gpu
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

    def test_can_use_accelerator_set_unset(self):
        # ensure we use CUB block reduction and not CUB device reduction
        old_routine_accelerators = _accelerator.get_routine_accelerators()
        _accelerator.set_routine_accelerators([])

        a = cupy.random.random((10, 10))
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
