import chainer
from chainer.functions.math import scale
from chainer import link
from chainer.links.connection import bias


class Scale(link.Chain):
    """Broadcasted elementwise product with learnable parameters.

    Computes a elementwise product as :func:`~chainer.functions.scale`
    function does except that its second input is a learnable weight parameter
    :math:`W` the link has.

    Args:
        axis (int): The first axis of the first input of
            :func:`~chainer.functions.scale` function along which its second
            input is applied.
        W_shape (tuple of ints): Shape of learnable weight parameter. If
            ``None``, this link does not have learnable weight parameter so an
            explicit weight needs to be given to its ``__call__`` method's
            second input.
        bias_term (bool): Whether to also learn a bias (equivalent to Scale
            link + Bias link).
        bias_shape (tuple of ints): Shape of learnable bias. If ``W_shape`` is
            ``None``, this should be given to determine the shape. Otherwise,
            the bias has the same shape ``W_shape`` with the weight parameter
            and ``bias_shape`` is ignored.

    .. seealso:: See :func:`~chainer.functions.scale` for details.

    Attributes:
        W (~chainer.Variable): Weight parameter if ``W_shape`` is given.
            Otherwise, no W attribute.
        bias (~chainer.links.Bias): Bias term if ``bias_term`` is ``True``.
            Otherwise, no bias attribute.

    """

    def __init__(self, axis=1, W_shape=None, bias_term=False, bias_shape=None):
        super(Scale, self).__init__()

        # Add W parameter and/or bias term.
        if W_shape is not None:
            self.add_param('W', W_shape)
            self.W.data.fill(1)
            if bias_term:
                func = bias.Bias(axis, W_shape)
                self.add_link('bias', func)
        else:
            if bias_term:
                if bias_shape is None:
                    raise ValueError('bias_shape should be given if W is not '
                                     'learnt parameter and bias_term is True.')
                func = bias.Bias(axis, bias_shape)
                self.add_link('bias', func)

        self.axis = axis

    def __call__(self, *xs):
        """Applies broadcasted elementwise product.

        Args:
            xs (list of Variables): Input variables whose length should
                be one if the link has a learnable weight parameter, otherwise
                should be two.
        """
        axis = self.axis

        # Case of only one argument where W is a learnt parameter.
        if hasattr(self, 'W'):
            if chainer.is_debug():
                assert len(xs) == 1
            x, = xs
            W = self.W
            z = scale.scale(x, W, axis)
        # Case of two arguments where W is given as an argument.
        else:
            if chainer.is_debug():
                assert len(xs) == 2
            x, y = xs
            z = scale.scale(x, y, axis)

        # Forward propagate bias term if given.
        if hasattr(self, 'bias'):
            return self.bias(z)
        else:
            return z
