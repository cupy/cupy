from chainer import function
from chainer import function_set
from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer.functions.connection import convolution_2d
from chainer.functions.pooling import max_pooling_2d


class Inception(function.Function):

    """Inception module of GoogLeNet.

    It applies four different functions to the input array and concatenates
    their outputs along the channel dimension. Three of them are 2D
    convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
    sizes have 1x1 convolutions (called projections) ahead of them. The other
    path consists of 1x1 convolution (projection) and 3x3 max pooling.

    The output array has the same spatial size as the input. In order to
    satisfy this, Inception module uses appropriate padding for each
    convolution and pooling.

    See: `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        in_channels (int): Number of channels of input arrays.
        out1 (int): Output size of 1x1 convolution path.
        proj3 (int): Projection size of 3x3 convolution path.
        out3 (int): Output size of 3x3 convolution path.
        proj5 (int): Projection size of 5x5 convolution path.
        out5 (int): Output size of 5x5 convolution path.
        proj_pool (int): Projection size of max pooling path.

    Returns:
        Variable: Output variable. Its array has the same spatial size and the
            same minibatch size as the input array. The channel dimension has
            size ``out1 + out3 + out5 + proj_pool``.

    .. note::

       This function inserts the full computation graph of the Inception module
       behind the input array. This function itself is not inserted into the
       computation graph.

    """

    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
        self.f = function_set.FunctionSet(
            conv1=convolution_2d.Convolution2D(in_channels, out1, 1),
            proj3=convolution_2d.Convolution2D(in_channels, proj3, 1),
            conv3=convolution_2d.Convolution2D(proj3, out3, 3, pad=1),
            proj5=convolution_2d.Convolution2D(in_channels, proj5, 1),
            conv5=convolution_2d.Convolution2D(proj5, out5, 5, pad=2),
            projp=convolution_2d.Convolution2D(in_channels, proj_pool, 1),
        )

    def __call__(self, x):
        out1 = self.f.conv1(x)
        out3 = self.f.conv3(relu.relu(self.f.proj3(x)))
        out5 = self.f.conv5(relu.relu(self.f.proj5(x)))
        pool = self.f.projp(max_pooling_2d.max_pooling_2d(
            x, 3, stride=1, pad=1))
        y = relu.relu(concat.concat((out1, out3, out5, pool), axis=1))
        return y

    # forward of Inception is a series of forward operations
    # of existing functions. So we do not need additional type checks.
    def check_type_forward(self, in_types):
        pass

    def to_gpu(self, device=None):
        return self.f.to_gpu(device)

    def to_cpu(self):
        return self.f.to_cpu()

    @property
    def parameters(self):
        return self.f.parameters

    @parameters.setter
    def parameters(self, params):
        self.f.parameters = params

    @property
    def gradients(self):
        return self.f.gradients

    @gradients.setter
    def gradients(self, grads):
        self.f.gradients = grads
