import unittest

import mock

import chainer
from chainer.functions.pooling import pooling_nd_kernel
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'ndim': [2, 3, 4],
}))
@attr.gpu
class TestPoolingNDKernelMemo(unittest.TestCase):

    def setUp(self):
        chainer.cuda.clear_memo()

    def test_pooling_nd_kernel_forward_memo(self):
        ndim = self.ndim
        with mock.patch('chainer.functions.pooling.pooling_nd_kernel.'
                        'PoolingNDKernelForward._generate') as m:
            pooling_nd_kernel.PoolingNDKernelForward.generate(ndim)
            m.assert_called_once_with(ndim)
            pooling_nd_kernel.PoolingNDKernelForward.generate(ndim)
            # Check that the mocked _generate() function is called just once
            # because the result of generate() function is cached.
            m.assert_called_once_with(ndim)

    def test_pooling_nd_kernel_backward_memo(self):
        ndim = self.ndim
        with mock.patch('chainer.functions.pooling.pooling_nd_kernel.'
                        'PoolingNDKernelBackward._generate') as m:
            pooling_nd_kernel.PoolingNDKernelBackward.generate(ndim)
            m.assert_called_once_with(ndim)
            pooling_nd_kernel.PoolingNDKernelBackward.generate(ndim)
            # Check that the mocked _generate() function is called just once
            # because the result of generate() function is cached.
            m.assert_called_once_with(ndim)


testing.run_module(__name__, __file__)
