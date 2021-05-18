import unittest
from unittest import mock

import cupy  # NOQA
from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


class CreateMock(object):

    def __init__(self, target):
        self.target = eval(target)
        self.retvals = []

    def __call__(self, *args, **kwargs):
        ret = self.target(*args, **kwargs)
        self.retvals.append(ret)
        return ret

    def check_number_of_ops(
            self, loops, memories, variables, lookup, mutate):
        # TODO(asi1024): Uncomment after replace fusion implementaiton.

        # assert isinstance(loops, int)
        # assert isinstance(memories, int)
        # assert isinstance(variables, int)
        # assert isinstance(lookup, list)
        # assert isinstance(mutate, list)
        # assert len(lookup) == len(mutate) == loops

        # assert len(self.retvals) == 1
        # history = self.retvals[0]
        # assert len(history.op_list) == loops
        # memory_space_set = set([p.memory for p in history.kernel_params])
        # assert len(memory_space_set) == memories
        # assert len(history.kernel_params) == variables
        # for op, r, w in zip(history.op_list, lookup, mutate):
        #     assert len(op.in_params) == r
        #     assert len(op.out_params) == w
        pass


def check_number_of_ops(
        loops, memories, variables, lookup, mutate):
    def wrapper(test_method):
        def new_impl(self, *args, **kwargs):
            target = 'cupy._core._fusion_trace.TraceImpl'
            with mock.patch(target, CreateMock(target)) as m:
                result = test_method(self, *args, **kwargs)
                m.check_number_of_ops(
                    loops, memories, variables, lookup, mutate)
            return result
        return new_impl
    return wrapper


@testing.gpu
class TestOptimizations(unittest.TestCase):

    def generate_inputs(self, xp):
        x = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    def generate_input_broadcast(self, xp):
        x = testing.shaped_random((3, 1, 4), xp, 'int64', scale=10, seed=0)
        y = testing.shaped_random((3, 5, 4), xp, 'int64', scale=10, seed=1)
        return (x, y), {}

    def generate_input_same_memory(self, xp):
        x = testing.shaped_random((4, 4), xp, 'int64', scale=10, seed=0)
        return (x, x, x.T), {}

    @check_number_of_ops(
        loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_elementwise_op(self, xp):
        return lambda x, y: x + y

    @check_number_of_ops(
        loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_op_1(self, xp):
        def impl(x, y):
            return x + x + y + y
        return impl

    @check_number_of_ops(
        loops=1, memories=3, variables=3, lookup=[2], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_op_2(self, xp):
        def impl(x, y):
            z = x + y
            return z + z
        return impl

    @check_number_of_ops(
        # TODO(asi1024): memory space = 3.
        loops=1, memories=4, variables=4, lookup=[3], mutate=[1])
    @fusion_utils.check_fusion()
    def test_fuse_elementwise_ops_4(self, xp):
        def impl(x, y):
            res = 0
            for i in range(10):
                res += x + y
            return res
        return impl

    @check_number_of_ops(
        loops=0, memories=1, variables=1, lookup=[], mutate=[])
    @fusion_utils.check_fusion()
    def test_ignore_op(self, xp):
        def impl(x, y):
            z = x + y  # NOQA
            return x
        return impl

    @check_number_of_ops(
        loops=1, memories=4, variables=4, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_returns_tuple(self, xp):
        def impl(x, y):
            return x + y, y - x
        return impl

    @check_number_of_ops(
        loops=2, memories=4, variables=6, lookup=[1, 3], mutate=[1, 1])
    @fusion_utils.check_fusion(
        generate_inputs_name='generate_input_broadcast')
    def test_different_shapes(self, xp):
        def impl(x, y):
            r1 = x + x
            r2 = x + y
            r3 = y + y
            return r1 * r2 * r3
        return impl

    @check_number_of_ops(
        loops=1, memories=2, variables=2, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_inplace_elementwise_1(self, xp):
        def impl(x, y):
            x += y
            y += x
            x += y
        return impl

    @check_number_of_ops(
        loops=1, memories=3, variables=3, lookup=[2], mutate=[2])
    @fusion_utils.check_fusion()
    def test_inplace_elementwise_2(self, xp):
        def impl(x, y):
            x += y
            return x + x
        return impl

    @check_number_of_ops(
        loops=1, memories=1, variables=1, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion(
        generate_inputs_name='generate_input_same_memory')
    def test_inplace_same_variable(self, xp):
        def impl(x, y, z):
            x += y
            x += y
        return impl

    @check_number_of_ops(
        loops=4, memories=3, variables=4,
        lookup=[1, 2, 1, 2], mutate=[1, 1, 1, 1])
    @fusion_utils.check_fusion(
        generate_inputs_name='generate_input_same_memory')
    def test_inplace_same_memory_space(self, xp):
        def impl(x, y, z):
            x += z
            x += z
        return impl

    @check_number_of_ops(
        loops=1, memories=2, variables=2, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_reduction_op(self, xp):
        return lambda x, y: xp.sum(x, axis=0)

    @check_number_of_ops(
        loops=1, memories=2, variables=3, lookup=[1], mutate=[1])
    @fusion_utils.check_fusion()
    def test_one_reduction_op_rotate(self, xp):
        return lambda x, y: xp.sum(x, axis=1)

    # TODO(asi1024): Fix parameters after optimization.
    @check_number_of_ops(
        loops=2, memories=4, variables=4, lookup=[2, 1], mutate=[1, 1])
    @fusion_utils.check_fusion()
    def test_one_fuse_reduction_premap(self, xp):
        def impl(x, y):
            premap = x + y
            return xp.sum(premap, axis=0)
        return impl

    # TODO(asi1024): Add tests for reduction.
