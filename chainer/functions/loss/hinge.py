import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _hinge_fwd_kernel():
    return cuda.elementwise(
        'S t', 'raw T bottom_diff',
        'int ind[] = {i, t}; bottom_diff[ind] *= -1',
        'hinge_fwd')


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
        num = len(x)
        self.bottom_diff = numpy.copy(x)
        self.bottom_diff[numpy.arange(num), t] *= -1
        self.bottom_diff = numpy.maximum(0, 1 + self.bottom_diff)
        if self.norm == 'L1':
            loss = self.bottom_diff.sum() / num
        elif self.norm == 'L2':
            loss = (self.bottom_diff ** 2).sum() / num
        else:
            raise NotImplementedError()

        return numpy.array(loss, dtype=x.dtype),

    def forward_gpu(self, inputs):
        x, t = inputs
        num = x.dtype.type(len(x))
        self.bottom_diff = cuda.cupy.maximum(
            0, 1 + _hinge_fwd_kernel()(t, x.copy()))
        if self.norm == 'L1':
            loss = self.bottom_diff.sum() / num
        elif self.norm == 'L2':
            loss = (self.bottom_diff ** 2).sum() / num
        else:
            raise NotImplementedError()

        return loss,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        self.bottom_diff[numpy.arange(len(t)), t] *= -1
        if self.norm == 'L1':
            gx = (gloss / len(t)) * numpy.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = (2 * gloss / len(t)) * self.bottom_diff
        else:
            raise NotImplementedError()

        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        t, gloss = inputs[1], grad_outputs[0]
        self.bottom_diff = _hinge_fwd_kernel()(t, self.bottom_diff)
        if self.norm == 'L1':
            gx = (gloss / len(t)) * xp.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = (2 * gloss / len(t)) * self.bottom_diff
        else:
            raise NotImplementedError()

        return gx, None


def hinge(x, t, norm='L1'):
    """Computes the hinge loss for a one-of-many classification task.

        .. math::
            L = \\frac{1}{N} \\sum_{n=1}^N \\sum_{k=1}^K \\left[
            \\max(0, 1 - \\delta\\{l_n = k\\} t_{nk}) \\right]^p

        where :math:`N` denotes the batch size, :math:`K` is the number of
        classes of interest,

        .. math::
            \\delta \\{ {\\rm condition} \\} = \\left \\{ \\begin{array}{cc}
            1 & {\\rm if~condition} \\\\
            -1 & {\\rm otherwise,}
            \\end{array} \\right.

        and

        .. math::
            p = \\left \\{ \\begin{array}{cc}
            1 & {\\rm if~norm} = {\\rm 'L1'} \\\\
            2 & {\\rm if~norm} = {\\rm 'L2'.}
            \\end{array} \\right.

    Args:
        x (~chainer.Variable): Input variable. The shape of ``x`` should be
            (:math:`N`, :math:`K`).
        t (~chainer.Variable): The :math:`N`-dimensional label vector
            :math:`{\\bf l}` with values
            :math:`l_n \in \{0, 1, 2, \dots, K-1\}`. The shape of ``t`` should
            be (:math:`N`,).
        norm (string): Specifies norm type. Only either 'L1' or 'L2' is
            acceptable.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            hinge loss :math:`L`.

    """
    return Hinge(norm)(x, t)
