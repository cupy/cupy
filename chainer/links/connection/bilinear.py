import numpy

from chainer.functions.connection import bilinear
from chainer import link


class Bilinear(link.Link):

    """Bilinear layer that performs tensor multiplication.

    Bilinear is a primitive link that wraps the
    :func:`~chainer.functions.bilinear` functions. It holds parameters ``W``,
    ``V1``, ``V2``, and ``b`` corresponding to the arguments of
    :func:`~chainer.functions.bilinear`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, parameters ``V1``, ``V2``, and ``b`` are
            omitted.
        initialW (3-D numpy array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
        initial_bias (tuple): Initial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.

    .. seealso:: See :func:`chainer.functions.bilinear` for details.

    Attributes:
        W (~chainer.Variable): Bilinear weight parameter.
        V1 (~chainer.Variable): Linear weight parameter for the first argument.
        V2 (~chainer.Variable): Linear weight parameter for the second
            argument.
        b (~chainer.Variable): Bias parameter.

    """
    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):
        super(Bilinear, self).__init__(W=(left_size, right_size, out_size))
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        if initialW is not None:
            assert initialW.shape == self.W.data.shape
            self.W.data[...] = initialW
        else:
            # TODO(Kenta OONO): I do not know appropriate way of
            # initializing weights in tensor network.
            # This initialization is a modification of
            # that of Linear function.
            in_size = left_size * right_size * out_size
            self.W.data[...] = numpy.random.normal(
                0, numpy.sqrt(1. / in_size), self.W.data.shape)

        if not self.nobias:
            self.add_param('V1', (left_size, out_size))
            self.add_param('V2', (right_size, out_size))
            self.add_param('b', out_size)

            if initial_bias is not None:
                V1, V2, b = initial_bias
                assert V1.shape == self.V1.data.shape
                assert V2.shape == self.V2.data.shape
                assert b.shape == self.b.data.shape
                self.V1.data[...] = V1
                self.V2.data[...] = V2
                self.b.data[...] = b
            else:
                self.V1.data[...] = numpy.random.normal(
                    0, numpy.sqrt(1. / left_size), (left_size, out_size))
                self.V2.data[...] = numpy.random.normal(
                    0, numpy.sqrt(1. / right_size), (right_size, out_size))
                self.b.data.fill(0)

    def __call__(self, e1, e2):
        """Applies the bilinear function to inputs and the internal parameters.

        Args:
            e1 (~chainer.Variable): Left input.
            e2 (~chainer.Variable): Right input.

        Returns:
            ~chainer.Variable: Output variable.

        """
        if self.nobias:
            return bilinear.bilinear(e1, e2, self.W)
        else:
            return bilinear.bilinear(e1, e2, self.W, self.V1, self.V2, self.b)

    def zero_grads(self):
        # Left for backward compatibility
        self.zerograds()
