import cupy as cp
from batchnorm.gen_input import gen_test_input
from batchnorm.nofuse import batchnorm_forward as nofuse_fwd
from batchnorm.fuse import batchnorm_forward as fuse_fwd
from batchnorm.elementwise import batchnorm_forward as elementwise_fwd

def assert_almost_equal(ret1, ret2, delta=1e-2):
    if type(ret1) != type(ret2):
        raise AssertionError('Type is different. ret1: {}, ret2: {}'.format(type(ret1), type(ret2)))
    if isinstance(ret1, tuple) or isinstance(ret1, list):
        for i in range(len(ret1)):
            assert_almost_equal(ret1[i], ret2[i])
        return
    if not isinstance(ret1, cp.core.core.ndarray):
        raise NotImplementedError('Check is not available for type {}'.format(ret1))
    dif = cp.sum(cp.abs(ret1 - ret2))
    if dif > delta:
        raise AssertionError('{} is not equal to {}.\n \
            Difference is {}.'.format(ret1, ret2, dif))

x = gen_test_input()

nofuse_ret = nofuse_fwd(x)
elementwise_ret = elementwise_fwd(x)
fuse_ret = fuse_fwd(x)

assert_almost_equal(nofuse_ret, elementwise_ret)
assert_almost_equal(nofuse_ret, fuse_ret)
