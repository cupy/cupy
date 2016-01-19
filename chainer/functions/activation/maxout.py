from chainer.functions.array import reshape
from chainer.functions.math import minmax


def maxout(x, num_channel, axis=1):
    """Maxout activation function

    It accepts an input tensor ``x``, reshapes the ``axis`` dimension
    (say the size being ``M * num_channel``) into two dimensional tensor
    ``(M, num_channel)`` and takes maximum along the third dimension.
    The output of this function is same as ``x`` except that ``axis`` dimension
    is transformed from ``M * channel`` to ``M``.

    Typically, ``x`` is the output of Linear Layer or Convolution Layer.
    The following is the example where we use :func:`maxout` in combination
    with Linear Link.

    >>> l = L.Linear(in_size, out_size * num_channel)
    ... x = Variable(...)  # prepare data
    ... x = linear(x)
    ... y = maxout(x, num_channel)

    Args:
       x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension.
    Returns:
        ~chainer.Variable: Variable holding :math:`Y`.

    .. seealso:: :class:`~chainer.links.Maxout`
    """

    assert (x.data.shape[axis] % num_channel == 0,
            'channel dimension must be divided by num_channel')
    shape = (x.data.shape[:axis] +
             (x.data.shape[axis] // num_channel, num_channel) +
             x.data.shape[axis + 1:])
    x = reshape.reshape(x, shape)
    return minmax.max(x, axis=axis + 1)
