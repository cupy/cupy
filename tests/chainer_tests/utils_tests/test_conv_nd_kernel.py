import unittest

import mock

import chainer
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv_nd_kernel


@testing.parameterize(*testing.product({
    'ndim': [2, 3, 4],
}))
@attr.gpu
class TestIm2colNDKernelMemo(unittest.TestCase):

    def setUp(self):
        chainer.cuda.clear_memo()

    def test_im2col_nd_kernel_memo(self):
        ndim = self.ndim
        with mock.patch(
                'chainer.utils.conv_nd_kernel.Im2colNDKernel._generate') as m:
            conv_nd_kernel.Im2colNDKernel.generate(ndim)
            m.assert_called_once_with(ndim)
            conv_nd_kernel.Im2colNDKernel.generate(ndim)
            m.assert_called_once_with(ndim)


@testing.parameterize(*testing.product({
    'ndim': [2, 3, 4],
}))
@attr.gpu
class TestCol2imNDKernelMemo(unittest.TestCase):

    def setUp(self):
        chainer.cuda.clear_memo()

    def test_col2im_nd_kernel_memo(self):
        ndim = self.ndim
        with mock.patch(
                'chainer.utils.conv_nd_kernel.Col2imNDKernel._generate') as m:
            conv_nd_kernel.Col2imNDKernel.generate(ndim)
            m.assert_called_once_with(ndim)
            conv_nd_kernel.Col2imNDKernel.generate(ndim)
            m.assert_called_once_with(ndim)


testing.run_module(__name__, __file__)
