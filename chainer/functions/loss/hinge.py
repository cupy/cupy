import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Hinge(function.Function):

    """Hinge loss."""

    def __init__(self, norm='L1'):
        if norm in ['L1', 'L2']:
            self.norm = norm
        else:
            raise NotImplementedError("norm should be either 'L1' or 'L2'")

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        num, dim = x.shape
        self.bottom_diff = numpy.copy(x)
        for i in six.moves.range(num):
            self.bottom_diff[i, t[i]] *= -1
        self.bottom_diff = max(0, 1 + self.bottom_diff)
        loss = 0
        if self.norm == 'L1':
            loss = numpy.sum(numpy.abs(self.bottom_diff)) / num
        elif self.norm == 'L2':
            loss = numpy.sum(self.bottom_diff ** 2) / num
        else:
            raise NotImplementedError()

        return numpy.array(loss, dtype=numpy.float32),

    def forward_gpu(self, inputs):
        x, t = inputs
        num, dim = x.shape
        self.bottom_diff = cuda.cupy.copy(x)
        self.bottom_diff = cuda.elementwise(
            'S t, int32 dim', 'raw T bottom_diff',
            'bottom_diff[i * dim + t] *= -1',
            'hinge_fwd')(t, dim, self.bottom_diff)
        self.bottom_diff = cuda.cupy.maximum(0, 1 + self.bottom_diff)
        loss = 0
        if self.norm == 'L1':
            loss = cuda.cupy.sum(cuda.cupy.abs(self.bottom_diff)) / num
        elif self.norm == 'L2':
            loss = cuda.cupy.sum(self.bottom_diff ** 2) / num
        else:
            raise NotImplementedError()

        return cuda.cupy.array(loss, dtype=cuda.cupy.float32),

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        for i in six.moves.range(t.shape[0]):
            self.bottom_diff[i, t[i]] *= -1
        if self.norm == 'L1':
            gx = (gloss / t.shape[0]) * numpy.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = (2 * gloss / t.shape[0]) * self.bottom_diff
        else:
            raise NotImplementedError()

        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        t, gloss = inputs[1], grad_outputs[0]
        self.bottom_diff = cuda.elementwise(
            'S t, int32 dim', 'raw T bottom_diff',
            'bottom_diff[i * dim + t] *= -1',
            'hinge_bwd')(t, inputs[0].shape[1], self.bottom_diff)
        if self.norm == 'L1':
            gx = (gloss / t.shape[0]) * xp.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = (2 * gloss / t.shape[0]) * self.bottom_diff
        else:
            raise NotImplementedError()

        return gx, None


def hinge(x, t, norm='L1'):
    """Computes the hinge loss for a one-of-many classification task.

        .. math::
            L = \\frac{1}{N} \\sum_{n=1}^N \\sum_{k=1}^K \\left[
            \\max(0, 1 - \\delta\\{l_n = k\\} t_{nk}) \\right]^p

        where :math:`N` denotes the batchsize, :math:`K` is the number of
        classes of interest,

        .. math::
            \\delta \\{ {\\rm condition} \\} = \\left \\{ \\begin{array}{cc}
            1 & {\\rm if~condition} \\\\
            -1 & {\\rm otherwise,}
            \\end{array} \\right.

        .. math::
            p = \\left \\{ \\begin{array}{cc}
            1 & {\\rm if~norm} = {\\rm 'L1'} \\\\
            2 & {\\rm if~norm} = {\\rm 'L2'.}
            \\end{array} \\right.

    Args:
        x (~chainer.Variable): Input variable. The shape of ``x`` should be
            (:math:`N`, :math:`K`).
        t (~chainer.Variable): The :math:`N`-dimensional label vector
            :math:`{\\bf l}` with values :math:`l_n \in [0, 1, 2, \dots, K-1]`.
            The shape of ``t`` should be (:math:`N`,).
        norm (string): Specifies norm type. Only either 'L1' or 'L2' is
            acceptable.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            hinge loss :math:`L`.

    """
    return Hinge(norm)(x, t)
