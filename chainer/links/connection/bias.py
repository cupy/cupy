import chainer
from chainer.functions.math import bias
from chainer import link


class Bias(link.Link):
    """Broadcasted elementwise summation with learnable parameters.

    Computes a elementwise summation as :func:`~chainer.functions.bias`
    function does except that its second input is a learnable bias parameter
    :math:`b` the link has.

    Args:
        axis (int): The first axis of the first input of
            :func:`~chainer.functions.bias` function along which its second
            input is applied.
        shape (tuple of ints): Shape of the learnable bias parameter. If
            ``None``, this link does not have learnable parameters so an
            explicit bias needs to be given to its ``__call__`` method's second
            input.

    .. seealso:: See :func:`~chainer.functions.bias` for details.

    Attributes:
        b (~chainer.Variable): Bias parameter if ``shape`` is given. Otherwise,
            no attributes.

    """

    def __init__(self, axis=1, shape=None):
        super(Bias, self).__init__()

        # Add b parameter if given.
        if shape is not None:
            self.add_param('b', shape)
            self.b.data.fill(0)

        self.axis = axis

    def __call__(self, *xs):
        """Applies broadcasted elementwise summation.

        Args:
            xs (list of Variables): Input variables whose length should
                be one if the link has a learnable bias parameter, otherwise
                should be two.
        """
        axis = self.axis

        # Case of only one argument where b is a learnt parameter.
        if hasattr(self, 'b'):
            if chainer.is_debug():
                assert len(xs) == 1
            x, = xs
            b = self.b
            return bias.bias(x, b, axis)
        # Case of two arguments where b is given as an argument.
        else:
            if chainer.is_debug():
                assert len(xs) == 2
            x, y = xs
            return bias.bias(x, y, axis)
